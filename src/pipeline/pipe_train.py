import pydoc
import json
import numpy as np
from loguru import logger

from src.data.dataset import CustomAudioDataset
from src.pipeline import PipelineGen
from sklearn.model_selection import train_test_split


class PipelineTrainer(PipelineGen):
    def __init__(self, config_path: str) -> None:
        # Config path
        self.config_path = config_path

        # load modules
        config_args = self.load_modules_from_json(config_path)

        # load gen attributes
        super().__init__(**config_args)
    
    def load_modules_from_json(self, json_path: str):
        config = json.load(open(json_path, 'r'))
        return self.load_modules_from_dict(config)

    def load_modules_from_dict(self, config: dict):    
        # Kwargs
        config_gen_args = {
            k: v 
            for k, v in config.items() 
            if k != 'modules'
        }

        # Loading class modules
        for module_name, module in config['modules'].items():
            class_ = pydoc.locate(f"src.{module['class_']}")
            params_ = module['params_']

            try:
                obj = class_(**params_)
                logger.debug(f"Module {module_name} successfully loaded")
            except Exception as e:
                logger.debug(f"Module {module_name} not loaded correctly." 
                             f"Please check the error:\n{e}")
                obj = None


            # assign to class attribute
            exec(f"self.{module_name} = obj")

        return config_gen_args
        
    def run_pipeline(self):
        # data loader
        df_info = self.loader.transform()

        # pre-processing
        audios, ind = self.preprocessor.transform_multifile(
            df_info['path']
        )

        # re-assignment
        df_info = df_info.loc[ind]

        # labels
        y = df_info[self.label_name].to_numpy()
        X = audios.copy()

        # label mapper
        label2id = {
            lbl: idx 
            for idx, lbl in enumerate(np.unique(y))
        }

        y = np.vectorize(label2id.get)(y)

        # Train, Val, Test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y,
            test_size=self.test_size,
            stratify=y
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, 
            y_train,
            train_size=self.train_size,
            stratify=y_train
        )

        # extractor
        ft_extractor = self.extractor.ft_extractor

        # extract features
        train_enc = self.extractor.transform(X_train)
        val_enc = self.extractor.transform(X_val)
        test_enc = self.extractor.transform(X_test)
        
        # datasets
        D_train = CustomAudioDataset(train_enc, y_train)
        D_val = CustomAudioDataset(val_enc, y_val)
        D_test = CustomAudioDataset(test_enc, y_test)

        # load model
        model = self.model.load_model(label2id)

        # trainer
        self.trainer.load_trainer(
            model=model, 
            train_dset=D_train, 
            val_dset=D_val, 
            tokenizer=ft_extractor,
        )

        # start training
        self.trainer.train()

        logger.debug('Completed')


    def export_artifacts(self, out_dir: str = "reports"):
        return None