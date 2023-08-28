import yaml


def read_yaml(path):
    """
    读取yaml类型配置文件数据
    :param path: yaml配置文件路径
    :return: 字典数据
    """
    with open(path, 'r', encoding='utf-8') as f:
        content = yaml.safe_load(f)
        return content
