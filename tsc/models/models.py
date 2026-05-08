import os
import inspect

def analyze(predictions, targets, class_names, fsize=-1):
    import torch
    from models.torch_running import Analyzer_Light, balanced_accuracy_score, f1_score
    if len(predictions.shape) > 1:
        predictions = torch.argmax(predictions, dim=1).cpu().numpy()
    metrics = {}
    analyzed = Analyzer_Light().analyze_classification(predictions, targets, class_names)
    metrics['accuracy'] = analyzed['total_accuracy']  # same as average recall over all classes
    metrics['precision'] = analyzed['prec_avg']  # average precision over all classes
    metrics['bal_acc'] = balanced_accuracy_score(targets, predictions)
    metrics['weighted_f1'] = f1_score(targets, predictions, average="weighted")
    metrics['micro_f1'] = f1_score(targets, predictions, average="micro")
    metrics['macro_f1'] = f1_score(targets, predictions, average="macro")
    metrics['file_size'] = fsize
    return metrics, None

def init_train(config, data):
    # torch-based models
    if config['model'] in ['SFCN', 'ConvTran', 'Quant', 'Hydra', 'Hydrant']:
        from models.torch_utils import init_torch
        init_torch(config)
        train_func = init_train_torch if config['model'] in ['SFCN', 'ConvTran'] else init_train_hydra_quant
        return train_func(config, data)
    # sktime models
    else:
        import sktime
        from sktime.registry import all_estimators
        SKT_MODEL_MAP = {mname: mclass for (mname, mclass) in all_estimators(estimator_types="classifier", return_names=True)}
        try:
            model_class = SKT_MODEL_MAP[config['model'] + 'Classifier'] # attach the 'Classifier' suffix for lookup
        except KeyError:
            raise NotImplementedError(f"Model {config['model']} not implemented.")
        config['software'] = 'sktime' + ' ' + sktime.__version__
        return init_train_sktime(config, data, model_class)
    
def init_train_sktime(config, data, model_class):
    model_args = {'random_state': config['seed'] if config['seed'] >= 0 else None}
    if 'n_jobs' in inspect.signature(model_class).parameters:
        model_args['n_jobs'] = -1 # use all available cores
    for arg in ['n_epochs', 'batch_size']:
        if arg in inspect.signature(model_class).parameters:
            model_args[arg] = config[arg]
    if 'optimizer' in inspect.signature(model_class).parameters:
        from keras.optimizers import Adam
        model_args['optimizer'] = Adam(learning_rate=config['lr'])
    model = model_class(**model_args)
    X_train, y_train = data['train_data'], data['train_label']
    
    def eval_func_creator(config, data, model):
        from sktime.utils import mlflow_sktime
        fsize = -1
        if config['use_pretrained']:
            model = mlflow_sktime.load_model(model_uri=os.path.join(config["use_pretrained"], 'sktime_model'))
            fsize = sum(d.stat().st_size for d in os.scandir(os.path.join(config["use_pretrained"], 'sktime_model')) if d.is_file())
        else:
            if not config['discard_model']:
                mlflow_sktime.save_model(sktime_model=model, path=os.path.join(config["output_dir"], 'sktime_model'))
                model = mlflow_sktime.load_model(model_uri=os.path.join(config["output_dir"], 'sktime_model'))
                fsize = sum(d.stat().st_size for d in os.scandir(os.path.join(config["output_dir"], 'sktime_model')) if d.is_file())
        return lambda: analyze(model._predict(data['test_data']), data['test_label'], config['labels'], fsize=fsize)

    return model, lambda: model.fit(X_train, y_train), eval_func_creator

def init_train_hydra_quant(config, data):
    # novel combination of Hydra and Quant that also internally supports pruning
    if config['model'] == 'Hydrant':
        from models.hydrant import Hydrant
        model = Hydrant(config)
    elif config['model'] == 'Hydra':
        if config['prune_rate'] > 0: # custom pruned Hydra variant
            from models.pruning import PrunedHydra
            model = PrunedHydra(config)
        else: # original Hydra, taken from https://github.com/angus924/aaltd2024
            from models.hydra import Hydra
            model = Hydra(config)
    elif config['model'] == 'Quant':
        if config['prune_rate'] > 0: # custom pruned Quant variant
            from models.pruning import PrunedQuant
            model = PrunedQuant(prune_rate=config['prune_rate'], prune_intermediate=config['prune_intermediate'], classifier=config['classifier'], num_estimators=config['num_estimators'], max_depth=config['max_depth'], max_features=config['max_features'], criterion=config['criterion'], seed=config['seed'])
        else: # original Quant, taken from https://github.com/angus924/aaltd2024
            from models.quant import QuantClassifier
            model = QuantClassifier(classifier=config['classifier'], num_estimators=config['num_estimators'], max_depth=config['max_depth'], max_features=config['max_features'], criterion=config['criterion'], seed=config['seed'])
        
    train_func = lambda: model.fit(data['train_data'], num_classes=config['n_labels']) # num classes required for hydra variants!

    def eval_func_creator(config, data, model):
        fsize = -1
        if config['use_pretrained']:
            fsize = model.load_from_disk(config["use_pretrained"])
        else:
            if not config['discard_model']:
                model.save_to_disk(config["output_dir"])
                fsize = model.load_from_disk(config["output_dir"])
        return lambda: analyze(model._predict(data['test_data'], num_classes=config['n_labels']), data['test_data'].Y, config['labels'], fsize=fsize) # TODO improve by only feeding X and batch_size info

    return model, train_func, eval_func_creator

def init_train_torch(config, data):
    from torch.utils.data import DataLoader
    from torch.optim import Adam
    from models.fcn import FCN
    from models.conv_tran import ConvTran
    from models.torch_utils import Torch_Dataset, get_loss_module, load_model
    from models.torch_running import SupervisedRunner, train_runner
    
    model = FCN(config) if config['model'] == 'SFCN' else ConvTran(config)
    train_loader = DataLoader(dataset=Torch_Dataset(data['train_data'], data['train_label']), batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=Torch_Dataset(data['val_data'], data['val_label']), batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    model.to(config['device'])
    optimizer = Adam(model.parameters(), lr=config['lr'])
    config['loss_module'] = get_loss_module()
    save_path = os.path.join(config['output_dir'], 'model_last.pth')

    trainer = SupervisedRunner(model, train_loader, config['device'], config['loss_module'], optimizer, l2_reg=0)
    val_evaluator = SupervisedRunner(model, val_loader, config['device'], config['loss_module'], optimizer, l2_reg=0)

    train_func = lambda: train_runner(config, model, trainer, val_evaluator, optimizer, save_path)

    def eval_func_creator(config, data, model):
        test_loader = DataLoader(dataset=Torch_Dataset(data['test_data'], data['test_label']), batch_size=config['batch_size'], shuffle=True, pin_memory=True)
        if config['use_pretrained']:
            save_path = os.path.join(config["use_pretrained"], 'model_last.pth')
        else:
            save_path = os.path.join(config['output_dir'], 'model_last.pth')
        best_model, _, _ = load_model(model, save_path)
        best_model.to(config['device'])
        fsize = os.path.getsize(save_path)

        return lambda: SupervisedRunner(best_model, test_loader, config['device'], config['loss_module']).evaluate(keep_all=True, fsize=fsize)

    return model, train_func, eval_func_creator
        