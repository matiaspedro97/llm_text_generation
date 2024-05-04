

class PipelineGen:
    def __init__(
            self,
            project_name: str,
            run_name: str,
            run_description: str,

            loader,
            model,
            trainer,
            **kwargs
    ) -> None:
        
        # Run details
        self.project_name = project_name
        self.run_name = run_name
        self.run_description = run_description

        # Modules
        self.loader = loader
        self.model = model
        self.trainer = trainer
