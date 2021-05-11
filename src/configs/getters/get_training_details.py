import collections
import sys
class Training_details():
    """
    class to get the training details
    """
    def __init__(self,file):
        self.file = file

    def _get_training_details(self):
        details_dic = collections.defaultdict(int)
        with open(self.file, "r") as details:
            for detail in details.readlines():
                if detail[0] not in ["#","\n"]:
                    split = detail.split()
                    details_dic[split[0]] = split[2] #works for this format #hardcoded
        return details_dic



