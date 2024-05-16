from utils import *
from modules import *
from data import *
from torch.utils.data import DataLoader
import torch.nn.functional as F
from datetime import datetime
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import torch.multiprocessing
import seaborn as sns
from pytorch_lightning.callbacks import ModelCheckpoint
import sys
import random
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
torch.multiprocessing.set_sharing_strategy('file_system')

def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "suim" or "directory":
    	return ['human divers', 'Wrecks/Ruins', 'Robots/ROVs/Instruments','Reefs/Invertebrates', 'Fish/Vertebrates/','Aquatic plants/Seagrass/Sea-floor/Rocks/Background']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))

class GraPix(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.dropout = torch.nn.Dropout2d(p=.1)
        self.proj_type = cfg.projection_type

        self.loss_ = []
        self.miou_ = []
        self.accu_ = []
        self.F1 = []

        if not cfg.continuous:
            self.dim = n_classes
        else:
            self.dim = cfg.dim

        data_dir = join(cfg.output_root, "data")

        if cfg.arch == "dino":
            self.net = DinoFeaturizer(self.dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))
        
        self.train_cluster_probe = GraphLookup(self.dim, n_classes)
        self.cluster_probe = GraphLookup(self.dim, n_classes + cfg.extra_clusters)

        # if cfg.confident_clustering:    
        #     #self.train_selfLabel_probe = SelfLabelLookup(self.cfg.threshold, self.cfg.apply_class_balancing)
        #     self.selfLabel_probe = SelfLabelLookup(self.cfg.threshold, self.cfg.apply_class_balancing)
 
        # elif cfg.knn_clustering:
        #     #self.train_knn_probe = KNNLookup(self.cfg.topk, self.cfg.entropy)
        #     self.knn_probe = KNNLookup(self.cfg.topk, self.cfg.entropy)

        arch = self.cfg.model_type
        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768

        self.cluster1 = self.make_clusterer(self.n_feats)
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

        #Decoder Layer
        self.decoder = nn.Conv2d(self.dim, self.net.n_feats, (1, 1))

        self.cluster_metrics = UnsupervisedMetrics(
            "test/cluster/", n_classes, self.cfg.extra_clusters, True)
        '''self.linear_metrics = UnsupervisedMetrics(
            "test/linear/", n_classes, 0, False)'''

        self.test_cluster_metrics = UnsupervisedMetrics(
            "final/cluster/", n_classes, self.cfg.extra_clusters, True)
        '''self.test_linear_metrics = UnsupervisedMetrics(
            "final/linear/", n_classes, 0, False)'''

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        elif self.cfg.dataset_name.startswith("directory"):
            self.label_cmap = create_suim_colormap()

        self.val_steps = 0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1))) 

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))
    
    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        #net_optim, linear_probe_optim, cluster_probe_optim = self.optimizers()
        net_optim, cluster_probe_optim = self.optimizers()

        net_optim.zero_grad()
        cluster_probe_optim.zero_grad()

        with torch.no_grad():
            ind = batch["ind"]
            img = batch["img"]
            img_aug = batch["img_aug"]
            coord_aug = batch["coord_aug"]
            #img_pos = batch["img_pos"]
            label = batch["label"]
            #label_pos = batch["label_pos"]

        feats = self.net(img)
        feats_aug = self.net(img_aug)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(feats))
            code_aug = self.cluster1(self.dropout(feats_aug))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(feats))
                code_aug += self.cluster2(self.dropout(feats_aug))
        else:
            code = feats
            code_aug = feats_aug

        log_args = dict(sync_dist=False, rank_zero_only=True)
        loss = 0
        
        if self.cfg.confident_clustering:
            #print('--------------------------------Self_label_training------------------------------------')
            img_products = torch.einsum("bchw,nc->bnhw", code, self.cluster_probe.clusters)
            aug_products = torch.einsum("bchw,nc->bnhw", code_aug, self.cluster_probe.clusters)
            cluster_loss = self.selfLabel_probe(img_products, aug_products, None)
        elif self.cfg.knn_clustering:
            #print('--------------------------------KNN_training------------------------------------')
            img_products = torch.einsum("bchw,nc->bnhw", code, self.cluster_probe.clusters)
            img_products = F.normalize(img_products,dim=1)
            code = F.normalize(code,dim=1)
            cluster_loss,_,_ = self.knn_probe(code, img_products)
        else:
            #print('--------------------------------Modularity_optimization--------------------------')
            cluster_loss,_, cluster_probs = self.cluster_probe(code, None)
         
        loss += cluster_loss
        self.log('loss/cluster', cluster_loss, **log_args)
        self.log('loss/total', loss, **log_args)

        self.manual_backward(loss)
        net_optim.step()
        cluster_probe_optim.step()
 
        if self.cfg.reset_probe_steps is not None and self.global_step == self.cfg.reset_probe_steps:
            print("RESETTING PROBES")
            self.cluster_probe.reset_parameters()
            self.trainer.optimizers[2] = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=5e-3)

        if self.global_step % 2000 == 0 and self.global_step > 0:
            print("RESETTING TFEVENT FILE")
            # Make a new tfevent file
            self.logger.experiment.close()
            self.logger.experiment._get_file_writer()

        #print(loss.item())
        self.loss_.append(loss.item())
        return loss

    def on_train_start(self):
        tb_metrics = {
            #**self.linear_metrics.compute(),
            **self.cluster_metrics.compute(),
        }
        print('------------------------{}---------------------'.format(tb_metrics))
        self.logger.log_hyperparams(self.cfg, tb_metrics)

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        self.outputs_ = []
        return
        
    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats = self.net(img)
            if self.proj_type is not None:
                code = self.cluster1(self.dropout(feats))
                if self.proj_type == "nonlinear":
                    code += self.cluster2(self.dropout(feats))
            else:
                code = feats
            
            cluster_loss, _, cluster_preds = self.cluster_probe(code, None)
            cluster_preds = cluster_preds.argmax(1).unsqueeze(1)
            cluster_pred = F.interpolate(cluster_preds.to(code.dtype), label.shape[-2:], mode='bilinear', align_corners=False)
            cluster_preds = cluster_pred.to(label.dtype)
            self.cluster_metrics.update(cluster_preds, label)
            
            self.outputs_.append( dict({
                'img': img[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}))
		
            return {
                'img': img[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()}

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        outputs = self.outputs_

        with torch.no_grad():
            tb_metrics = {
                **self.cluster_metrics.compute(),
            }

            for metric, value in tb_metrics.items():
                if metric == 'test/cluster/mIoU':
                    self.miou_.append(value)
                elif metric == 'test/cluster/Accuracy':
                    self.accu_.append(value)
                else:
                    self.F1.append(value)
   
            if self.global_step > 2:
                self.log_dict(tb_metrics)

                if self.trainer.is_global_zero and self.cfg.azureml_logging:
                    from azureml.core.run import Run
                    run_logger = Run.get_context()
                    
                    for metric, value in tb_metrics.items():
                        run_logger.log(metric, value)

            self.cluster_metrics.reset()

    def configure_optimizers(self):
        main_params = list(self.net.parameters())
        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=1e-3)#, weight_decay= 5e-3)
        return net_optim, cluster_probe_optim

    def save_plot(self,type_, data_):
        plt.plot(np.arange(1, len(data_)+1), data_, label=type_)
        plt.xlabel('Epoch')
        plt.ylabel(type_)
        plt.legend()
        plt.savefig('plots/{}_{}_plot.png'.format(self.cfg.experiment_name ,type_))
        plt.close()


@hydra.main(config_path="configs", config_name="train.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")

    prefix = "{}/{}_{}".format(cfg.log_dir, cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    random.seed(0)

    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((5, 5))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=False,
        pos_labels=False
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)

    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        val_batch_size = cfg.batch_size

    val_loader = DataLoader(val_dataset, val_batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    
    if cfg.confident_clustering:
        model =  GraPix.load_from_checkpoint(cfg.model_paths[0], cfg = cfg, strict=False)
        print("-----------------------------Loading pretrained weight's to the model-----------------------------------------")
    elif cfg.knn_clustering:
        model =  GraPix.load_from_checkpoint(cfg.model_paths_1[0], cfg = cfg, strict=False)
        print("-----------------------------Loading pretrained weight's to the model-----------------------------------------")
    else:
        model = GraPix(train_dataset.n_classes, cfg)
 
    tb_logger = TensorBoardLogger(
        join(log_dir, name),
        default_hp_metric=False
    )

    if cfg.submitting_to_aml:
        gpu_args = dict(gpus=1, val_check_interval=250)
        if gpu_args["val_check_interval"] > len(train_loader):
            gpu_args.pop("val_check_interval")

    else:
        gpu_args = dict(devices=-1, accelerator='auto', val_check_interval=cfg.val_freq)
        if gpu_args["val_check_interval"] > len(train_loader) // 4:
            gpu_args.pop("val_check_interval")

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=tb_logger,
        max_steps=cfg.max_steps,
        callbacks=[
            ModelCheckpoint(
                dirpath=join(checkpoint_dir, name),
                every_n_train_steps=400,
                save_top_k=2,
                monitor="test/cluster/mIoU",
                mode="max",
                save_last=True,
            )
        ],
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)

    #print(model.loss_)
    model.save_plot('Loss',model.loss_)
    model.save_plot('MiOU',model.miou_)
    model.save_plot('Accuracy',model.accu_)
    model.save_plot('F1Score',model.F1)
    print('Maximum MioU:', max(model.miou_[1:]))

    if cfg.confident_clustering:
        model.save_plot('True_sample', model.selfLabel_probe.true_samples)
        model.save_plot('False_sample', model.selfLabel_probe.false_samples)

if __name__ == "__main__":
    prep_args()
    my_app()