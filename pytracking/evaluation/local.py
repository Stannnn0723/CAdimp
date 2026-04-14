from pytracking.evaluation.environment import EnvSettings


def local_env_settings():
    settings = EnvSettings()

    # 网络权重存放路径（推理时模型读取位置）
    settings.network_path = '/data/lyx/project/pytracking-master/pytracking/networks/'

    # 跟踪结果存放路径
    settings.results_path = '/data/lyx/project/pytracking-master/tracking_results/'

    # 数据集路径（评测时使用）
    settings.otb_path = ''
    settings.nfs_path = ''
    settings.uav_path = ''
    settings.tpl_path = ''
    settings.uav_path = r'/data/lyx/dataset/UAV123/UAV123/'

    settings.vot_path = ''
    settings.got10k_path = ''
    settings.lasot_path = ''
    settings.lasot_extension_subset_path = ''
    settings.trackingnet_path = ''
    settings.oxuva_path = ''
    settings.davis_dir = ''
    settings.youtubevos_dir = ''

    # GOT-10k 打包结果路径
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''

    # TrackingNet 打包结果路径
    settings.tn_packed_results_path = ''
    
    settings.latot_path = r'/data/lyx/dataset/LaTOT/LaTOT/'

    settings.small112_path = r'/data/lyx/dataset/small112/'
    settings.small90_path = r'/data/lyx/dataset/small90/small90/'

    settings.dtb70_path = r'/data/lyx/dataset/DTB70/DTB70/'

    return settings
