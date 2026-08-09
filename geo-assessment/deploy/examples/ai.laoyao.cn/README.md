# ai.laoyao.cn 示例

这个目录记录官方演示站点 <https://ai.laoyao.cn/geo/> 的部署参数，方便维护者复现升级。通用部署请使用上一级的 `install.sh` 与 `nginx-subdirectory.conf.example`。

项目路径为 `/www/wwwroot/ai.laoyao.cn/geo`，持久化数据目录为 `/www/wwwroot/ai.laoyao.cn/geo-data`，运行用户为 `www:www`，PHP 命令为 `/www/server/php/73/bin/php`，PHP-FPM socket 为 `/tmp/php-cgi-73.sock`。版本升级时复用该数据目录，数据库、密钥、日志和备份会继续保留。

首次从旧版 `geo/storage/` 切换时，先创建并验证备份，暂停应用写入，再将该目录的全部内容复制到 `/www/wwwroot/ai.laoyao.cn/geo-data/`。安装脚本检测到旧数据且新目录为空时会停止，防止误建空数据库。

```bash
cd /www/wwwroot/ai.laoyao.cn/geo
bash deploy/examples/ai.laoyao.cn/install.sh
/www/server/nginx/sbin/nginx -t
```

`nginx.conf` 启用官方演示站点的百度统计 ID。统计脚本只在未登录首页加载，身份、答题、报告与证书页面保持关闭。演示首页需要同步披露统计用途；其他部署者应替换成自己的 ID，或删除该参数保持统计关闭。
