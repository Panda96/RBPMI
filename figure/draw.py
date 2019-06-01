# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
labels = ["blank", "parse error", "overlapped pools", "flows cross pools", "unmatched"]
sizes = [6, 90, 1259, 261, 67]
plt.pie(sizes, labels=labels)
plt.axis("equal")
plt.show()
plt.legend
