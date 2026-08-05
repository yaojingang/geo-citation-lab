# ai.laoyao.cn 示例

这个目录记录官方演示站点 <https://ai.laoyao.cn/geo/> 的部署参数，方便维护者复现升级。通用部署请使用上一级的 `install.sh` 与 `nginx-subdirectory.conf.example`。

项目路径为 `/www/wwwroot/ai.laoyao.cn/geo`，运行用户为 `www:www`，PHP 命令为 `/www/server/php/73/bin/php`，PHP-FPM socket 为 `/tmp/php-cgi-73.sock`。

```bash
cd /www/wwwroot/ai.laoyao.cn/geo
bash deploy/examples/ai.laoyao.cn/install.sh
/www/server/nginx/sbin/nginx -t
```

`nginx.conf` 启用官方演示站点的百度统计 ID。演示首页需要同步披露统计用途；其他部署者应替换成自己的 ID，或删除该参数保持统计关闭。
