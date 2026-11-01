import os
import cv2
import matplotlib.pyplot as plt
from typing import Optional


class Logger:
    """ A class to encapsulate external loggers such as Tensorboard, Allegro ClearML, Neptune, Weight and Biases,
        Comet, ...
    """
    __main_logger = None  # type: Optional[Logger]

    @classmethod
    def current_logger(cls):
        # type: () -> Logger
        return cls.__main_logger

    def __init__(self, cfg):
        # self.cfg = cfg
        # self.model_name = cfg.project.start_time + cfg.project.experiment_id

        # configs
        self.save_disk = cfg.project.logger.save_disk
        self.save_dir = cfg.data.save_dir
        self.matplotlib_show = cfg.project.logger.matplotlib_show

        # init external loggers
        self.tensorboard_logger = None
        if cfg.project.logger.use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_folder = os.path.join(cfg.data.save_dir, 'tensorboard')
            self.tensorboard_logger = SummaryWriter(self.tensorboard_folder)

        self.allegro_clearml_task = None
        if cfg.project.logger.use_clearml:
            from clearml import Task
            # Automagic Allegro clearml Task setup
            self.allegro_clearml_task = Task.init(
                project_name=cfg.project.name,
                task_name=cfg.project.experiment_name,
                task_type=Task.TaskTypes.training,
                reuse_last_task_id=False,
                continue_last_task=False,
                output_uri=None,
                auto_connect_arg_parser=True,
                auto_connect_frameworks=True,
                auto_resource_monitoring=True
            )
            # Register hyper-parameters
            self.allegro_clearml_task.connect(cfg)

        self.use_neptune = cfg.project.logger.use_neptune
        if self.use_neptune:
            global neptune
            import neptune
            from pandas import json_normalize
            nested_to_record = lambda value, sep: json_normalize(dict(value), sep=sep).to_dict(orient="records")[0]
            project_qualified_name = os.environ.get("NEPTUNE_PROJECT")
            api_token = os.environ.get("NEPTUNE_API_TOKEN")
            assert api_token is not None and project_qualified_name is not None, "Set NEPTUNE_PROJECT and NEPTUNE_API_TOKEN to enable Neptune"
            neptune.init(project_qualified_name=project_qualified_name.format(cfg.project.name),
                         api_token=api_token,
                         )
            # Create experiment
            neptune.create_experiment(params=nested_to_record(cfg, sep='.'))

        self.use_wandb = cfg.project.logger.use_wandb
        if self.use_wandb:
            global wandb
            import wandb
            # os.environ["WANDB_SILENT"] = "true"
            # wandb.init(config=cfg, sync_tensorboard=True, project=cfg.project.name, dir=cfg.data.save_dir, reinit=False)
            if cfg.project.logger.use_tensorboard:
                wandb.tensorboard.patch(pytorch=True, save=True, root_logdir=self.tensorboard_folder)
            wandb.init(config=cfg,
                       project=cfg.project.name,
                       dir=cfg.data.save_dir,
                       reinit=False,
                       name=str(cfg.project.job_id),
                       notes=cfg.project.notes,
                       tags=cfg.project.tags
                       )
            # wandb.tensorboard.patch(save=True, tensorboardX=False)

        Logger.__main_logger = self

    def add_model(self, model):
        if self.use_wandb and wandb.run is not None:
            wandb.watch(model)

    def add_text(self, tag, value):
        if self.use_wandb and wandb.run is not None:
            wandb.log({tag: value})
        if self.use_neptune:
            neptune.log_text(tag, 0, value)

    def add_scalar(self, tag, scalar_value, step):
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.add_scalar(tag, scalar_value, step)
        if self.use_wandb and wandb.run is not None:
            wandb.log({tag: scalar_value})
        if self.use_neptune:
            neptune.log_metric(tag, x=step, y=scalar_value)

    def add_figure(self, tag, figure, step):
        if self.matplotlib_show:
            figure.show()
            plt.waitforbuttonpress()
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.add_figure(tag, figure, step)
        if self.allegro_clearml_task is not None:
            self.allegro_clearml_task.logger.report_matplotlib_figure(
                title=tag,
                # series="Iteration {:06d}".format(self.global_step()),
                series="",
                iteration=step,
                figure=figure,
                report_image=False
            )
        if self.use_wandb and wandb.run is not None:
            wandb.log({tag: wandb.Image(
                figure)})  # FIXME cannot give "figure" directly: Invalid value of type 'builtins.str' received for the 'color' property of scatter.marker Received value: 'none'
        if self.use_neptune:
            neptune.log_image(tag, figure)
            # neptunecontrib.api.log_chart(tag, figure) # FIXME Invalid value of type 'builtins.str' received for the 'color' property of scatter.marker Received value: 'none'
        if self.save_disk:
            figure_path = os.path.join(self.save_dir, 'figures', tag + '.png')
            os.makedirs(os.path.dirname(figure_path), exist_ok=True)
            plt.savefig(figure_path)
        plt.close(figure)

    def add_image(self, group, name, image, step):
        """Input image must be in RGB format"""
        # if self.tensorboard_logger is not None:
        #     self.tensorboard_logger.add_figure(tag, figure, self.global_step())
        if self.allegro_clearml_task is not None:
            self.allegro_clearml_task.logger.report_image(
                title=group,
                series=name,
                iteration=step,
                # local_path=None,
                image=image,
                # matrix=None,
                max_image_history=50,
                # delete_after_upload=False,
                # url=None
            )

        if self.use_wandb and wandb.run is not None:
            wandb.log({group + name: wandb.Image(image)})
        if self.use_neptune:
            neptune.log_image(group + name, image)
        if self.save_disk:
            image_path = os.path.join(self.save_dir, 'images', f"{group}_{name}.jpg")
            os.makedirs(os.path.dirname(image_path), exist_ok=True)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(image_path, image)

    def add_embeddings(self, tag, embeddings, labels, imgs, step):
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.add_embedding(embeddings,
                                                  metadata=labels,
                                                  label_img=imgs,
                                                  global_step=step,
                                                  tag=tag,
                                                  metadata_header=None)

    def close(self):
        if self.tensorboard_logger is not None:
            self.tensorboard_logger.close()
            if self.allegro_clearml_task is not None:
                self.allegro_clearml_task.upload_artifact('Tensorboard folder', artifact_object=self.tensorboard_folder)
                self.allegro_clearml_task.close()
        if self.use_wandb and wandb.run is not None:
            wandb.finish()
