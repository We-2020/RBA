#!/bin/bash
###
 # @Author: snxy 1113885219@qq.com
 # @Date: 2025-12-06 17:09:44
 # @LastEditors: snxy 1113885219@qq.com
 # @LastEditTime: 2025-12-17 08:22:28
 # @FilePath: /caojiaxiang/brain age/RBA/gitUpdate.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
### 

###########################################
# 填写你的当前模型指标
MAE=2.66
VAL_RBA=5.10
###########################################

DATE=$(date +"%Y%m%d")
TAG="mae${MAE}_rba${VAL_RBA}_${DATE}"
MSG="Update: MAE=${MAE}, RBA=${VAL_RBA}"

echo "==> Adding files"
git add .

echo "==> Committing: $MSG"
git commit -m "$MSG"

echo "==> Creating tag: $TAG"
git tag -a "$TAG" -m "MAE: $MAE | Val_RBA: $VAL_RBA"

echo "==> Pushing commit and tag"
git push
git push origin "$TAG"

echo "==> Done! Created tag: $TAG"
