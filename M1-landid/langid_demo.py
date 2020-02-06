#encoding=utf-8

import langid

s1 = "中国"
result= langid.classify(s1)
print (result)

#result:('zh', -17.446399450302124)