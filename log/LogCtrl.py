import logging
import shutil

class Misakalog:
    def __init__(self, config):
        # 首先保存当前的config文件
        logname = "./log/"+"log_of_misaka_"+config.MisakaNum+'-'+str(config.step)
        try:
            f1 = open("config.py", encoding="utf-8")
            f2 = open(logname, "a", encoding="utf-8")
            f2.write("\n\n********************************** the config of exeperience ********************************** \n\n")
            shutil.copyfileobj(f1, f2)
            f2.write("\n\n*********************************************************************************************** \n\n")
        finally:
            if(f1):
                f1.close()
            if (f2):
                f2.close()

        self.logg = logging.getLogger("log_of_misaka_"+config.MisakaNum)
        self.logg.handlers = []
        # 定义一个模板
        self.FORMATTER = logging.Formatter("%(asctime)s-%(message)s")
        # 创建一个屏幕流
        self.p_stream = logging.StreamHandler()
        # 创建一个文件流
        self.f_stream = logging.FileHandler(logname, mode="a", encoding="utf-8")
        # 将流绑定到模板
        self.p_stream.setFormatter(self.FORMATTER)
        self.f_stream.setFormatter(self.FORMATTER)
        # 将日志和流进行绑定
        self.logg.addHandler(self.p_stream)
        self.logg.addHandler(self.f_stream)
        # 设置日志记录等级
        self.logg.setLevel(logging.DEBUG)

    def printInfo(self, str):
        self.logg.info(str)
        # logg.warning("this is warning")
        # logg.error("this is error")
        # logg.critical("this is critical")




