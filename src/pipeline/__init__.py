

class PipelineGen:
    def __init__(
            self, 
            run_id: int, 
            run_description: str, 
            label_name: str,
            path_name: str,
            model_name: str,
            group_name: str,
            train_size: float,
            test_size: float,
            **kwargs
    ) -> None:
        
        # Run details
        self.run_id = run_id
        self.run_description = run_description

        # Model name
        self.model_name = model_name
        
        # Column names
        self.label_name = label_name
        self.path_name = path_name
        self.group_name = group_name

        # Training settings
        self.train_size = train_size
        self.test_size = test_size