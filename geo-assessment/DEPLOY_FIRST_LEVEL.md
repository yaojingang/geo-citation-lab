# ai.laoyao.cn/geo 部署说明

这份部署说明与专用压缩包对应：

- 线上地址：`https://ai.laoyao.cn/geo/`
- 服务器目录：`/www/wwwroot/ai.laoyao.cn/geo`
- 应用基础路径：`/geo`
- 站点环境：宝塔目录结构、Nginx/Tengine
- PHP 要求：PHP 7.3.5 或更高版本，并启用 `json`、`mbstring`、`openssl`、`pdo`、`pdo_sqlite`
- SQLite 要求：PDO 实际连接的 SQLite 3.24.0 或更高版本

PHP 7.3 已结束官方安全维护。本包用于当前服务器兼容上线；具备维护窗口后，建议把该站点 FPM 升级到仍受安全支持的 PHP 版本。应用源码同时兼容 PHP 7.3.5–8.4。

当前站点已通过 `aid.laoyao.cn.conf` 及其扩展配置提供 `/geo` 服务。新版静态资源带文件版本参数，覆盖更新后浏览器会自动获取最新样式和脚本。

## 一、上传并解压

把 `geo-assessment-ui-v3-inline-analytics-php73-sqlite324-ai-laoyao-cn-geo-20260805.zip` 上传到 `/www/wwwroot/ai.laoyao.cn/`，然后执行：

```bash
cd /www/wwwroot/ai.laoyao.cn
unzip geo-assessment-ui-v3-inline-analytics-php73-sqlite324-ai-laoyao-cn-geo-20260805.zip
cd /www/wwwroot/ai.laoyao.cn/geo
```

压缩包的顶层目录已经固定为 `geo`。压缩包不含本地测试数据库、应用密钥、日志、备份和开发依赖。

## 二、初始化应用

宝塔 PHP-FPM 通常使用 `www` 用户。项目内附带初始化脚本：

```bash
cd /www/wwwroot/ai.laoyao.cn/geo
sudo bash deploy/install-ai-laoyao-cn-geo.sh
```

脚本会检查项目路径、PHP 版本、必需扩展和运行用户，然后设置 `storage/` 权限，并依次执行：

```bash
/www/server/php/73/bin/php bin/console app:install
/www/server/php/73/bin/php bin/console questions:verify
/www/server/php/73/bin/php bin/console app:health
```

安装脚本会优先使用宝塔 PHP 7.3 的 `/www/server/php/73/bin/php`。该文件不存在时，才会使用系统 `PATH` 中的 `php`。脚本结束时会打印实际采用的 PHP 命令、PHP 版本和 PDO SQLite 版本。

若 PHP-FPM 使用其他用户，可显式指定：

```bash
sudo WEB_USER=www-data bash deploy/install-ai-laoyao-cn-geo.sh
```

若 PHP 7.3 安装在其他路径，可同时指定命令：

```bash
sudo env PHP_BIN=/实际路径/php WEB_USER=www-data bash deploy/install-ai-laoyao-cn-geo.sh
```

`app:install` 会生成 `storage/app.key`、创建 `storage/app.sqlite`、执行迁移并导入 30 道题。重复执行会保留已有密钥和数据。

查看 PHP 7.3 实际连接的 SQLite 版本：

```bash
/www/server/php/73/bin/php -r '$pdo=new PDO("sqlite::memory:"); echo "PHP=",PHP_VERSION,PHP_EOL,"SQLite=",$pdo->query("SELECT sqlite_version()")->fetchColumn(),PHP_EOL;'
```

安装脚本会在写入数据库前检查 SQLite 版本。这里需要查看 PDO 的查询结果，系统 `sqlite3 --version` 可能指向另一套运行库。

## 三、确认 PHP-FPM socket

先查看当前站点或宝塔 PHP 配置使用的 socket：

```bash
grep -R "fastcgi_pass" \
  /www/server/panel/vhost/nginx/aid.laoyao.cn.conf \
  /www/server/panel/vhost/nginx/extension/aid.laoyao.cn/geo-assessment.conf \
  /www/server/nginx/conf/enable-php-*.conf 2>/dev/null | head -20
```

当前服务器 PHP 7.3 的宝塔常见值是：

```nginx
fastcgi_pass unix:/tmp/php-cgi-73.sock;
```

如果检查结果不同，请修改 `deploy/nginx-ai.laoyao.cn-geo.conf` 中的 `fastcgi_pass`。应用需要 PHP 7.3.5 或更高版本。

## 四、配置 Nginx/Tengine

先备份站点配置：

```bash
sudo cp \
  /www/server/panel/vhost/nginx/aid.laoyao.cn.conf \
  /www/server/panel/vhost/nginx/aid.laoyao.cn.conf.backup-20260805
```

当前服务器的主配置是 `/www/server/panel/vhost/nginx/aid.laoyao.cn.conf`，`/geo` 规则保存在 `/www/server/panel/vhost/nginx/extension/aid.laoyao.cn/geo-assessment.conf`。首次安装时可把 `deploy/nginx-ai.laoyao.cn-geo.conf` 复制到该扩展文件；覆盖升级无需再次修改 Nginx 配置。

专用配置如下：

```nginx
location = /geo {
    return 301 /geo/;
}

location ^~ /geo/assets/ {
    alias /www/wwwroot/ai.laoyao.cn/geo/public/assets/;
    access_log off;
    expires 7d;
}

location = /geo/index.php {
    include /www/server/nginx/conf/fastcgi.conf;
    fastcgi_param SCRIPT_FILENAME /www/wwwroot/ai.laoyao.cn/geo/public/index.php;
    fastcgi_param SCRIPT_NAME /geo/index.php;
    fastcgi_param APP_BASE_PATH /geo;
    fastcgi_param APP_ENV production;
    fastcgi_param APP_DEBUG 0;
    fastcgi_pass unix:/tmp/php-cgi-73.sock;
}

location ^~ /geo/ {
    rewrite ^ /geo/index.php last;
}
```

配置测试通过后再重载：

```bash
sudo /www/server/nginx/sbin/nginx -t
sudo /etc/init.d/nginx reload
```

如果服务器由 systemd 管理 Nginx，可使用：

```bash
sudo systemctl reload nginx
```

任何配置测试错误都应先修正，此时不要执行 reload。

## 五、上线验证

```bash
curl -I https://ai.laoyao.cn/geo
curl -fsS https://ai.laoyao.cn/geo/ >/dev/null
curl -fsS https://ai.laoyao.cn/geo/ \
  | grep 'https://hm.baidu.com/hm.js?c0aa4d814a9bb0449f84d59c73cc5da4'
```

第一条应返回到 `/geo/` 的 301，第二条应成功返回页面，第三条应在页面内匹配百度统计初始化代码。随后在浏览器完成以下验证：

1. 首页样式和百度统计脚本正常加载
2. 输入姓名后可以进入第 1 题
3. 单选、多选、上一题、下一题和题号跳转可以保存
4. 交卷后报告图表和 30 题解析正常显示
5. 刷新浏览器后登录态与历史记录仍然存在

## 六、更新与备份

更新前执行：

```bash
cd /www/wwwroot/ai.laoyao.cn/geo
sudo -u www /www/server/php/73/bin/php bin/console backup:create
sudo -u www /www/server/php/73/bin/php bin/console backup:verify
```

确认备份通过后，把新版压缩包放到 `/www/wwwroot/ai.laoyao.cn/`，执行 `unzip -o` 覆盖应用文件，再进入 `geo` 目录重新运行安装脚本和健康检查。发布包仅包含 `storage/.gitignore`，不会覆盖当前 `storage/app.sqlite`、`storage/app.key`、`storage/logs/` 和 `storage/backups/`。

## 七、配置回滚

如果新增 location 后站点异常，恢复备份并验证：

```bash
sudo cp \
  /www/server/panel/vhost/nginx/aid.laoyao.cn.conf.backup-20260805 \
  /www/server/panel/vhost/nginx/aid.laoyao.cn.conf
sudo /www/server/nginx/sbin/nginx -t
sudo /etc/init.d/nginx reload
```

应用目录和 SQLite 数据可继续保留，方便排查或再次上线。

## 常见问题

- `/geo/` 返回 502：核对 `fastcgi_pass` 指向 `/tmp/php-cgi-73.sock`，并确认 PHP 7.3 FPM 正常运行
- `/geo/` 返回 404：确认四个 `/geo` location 位于正确的 `server {}`，且优先于现有通用代理规则
- 页面没有样式：确认静态资源 alias 和 `APP_BASE_PATH /geo` 完全一致
- 页面返回 503：以 `www` 用户运行 `/www/server/php/73/bin/php bin/console app:health`，检查 PHP 扩展和 `storage/` 权限
- SQLite 无法写入：确认 PHP-FPM 用户拥有整个 `storage/` 目录的读写权限
- SQLite 版本不足：运行上面的 PDO 检查命令，确认版本达到 3.24.0
- HTTPS 登录态异常：受信反向代理场景可设置 `GEO_TRUST_PROXY=1`
