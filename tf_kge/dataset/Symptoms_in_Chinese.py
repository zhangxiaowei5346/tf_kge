# coding=utf-8
from tf_kge.dataset.common import CommonDataset


class Symptoms_in_ChineseDataset(CommonDataset):
    def __init__(self, dataset_root_path=None):
        super().__init__(dataset_name="Symptoms_in_Chinese",
                         download_urls=[
                             "https://github.com/zhangxiaowei5346/TKDE_DATA/raw/main/Symptoms-in-Chinese.zip"
                         ],
                         download_file_name="Symptoms-in-Chinese.zip",
                         cache_name=None,#"cache.p",
                         dataset_root_path=dataset_root_path)

