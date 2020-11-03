# coding=utf-8
from tf_kge.dataset.common import CommonDataset


class FB15k237Dataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="FB15k-237",
                         download_urls=[
                             "https://github.com/zhangxiaowei5346/TKDE_DATA/raw/main/FB15k-237.zip"
                         ],
                         download_file_name="FB15k-237.zip",
                         cache_name=None,#"cache.p",
                         dataset_root_path=dataset_root_path)

