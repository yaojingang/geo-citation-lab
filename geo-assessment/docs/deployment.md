# 部署说明

## 构建与首次安装

```bash
composer install --no-dev --classmap-authoritative
sudo GEO_DATA_DIR=/srv/geo-assessment/shared GEO_PUBLIC_URL=https://geo.example.com WEB_USER=www-data PHP_BIN=php bash deploy/install.sh
```

站点文档根必须设为项目的 `public/`。将整个项目目录暴露为文档根会增加数据库和密钥泄露风险。

GitHub Release 压缩包已经包含生产依赖，服务器可以直接解压后运行 `deploy/install.sh`。从源码部署时先运行 Composer。安装脚本要求 root 权限，`WEB_USER` 应与 PHP-FPM 进程用户一致，`PHP_BIN` 可以填写完整路径。

`GEO_DATA_DIR` 保存数据库、应用密钥、日志和备份。生产部署应使用版本目录外的固定绝对路径，并在 PHP-FPM 或 Web 服务器配置中传入同一个值。安装脚本会创建目录并验证运行用户权限。

已有版本仍把数据保存在版本目录内时，先用当前版本创建并验证备份，暂停应用写入，将当前 `storage/` 的全部内容完整复制到共享目录，再运行新版本安装脚本。首次运行时可增加 `GEO_LEGACY_DATA_DIR=/srv/geo-assessment/current/storage`；目标共享目录为空且旧数据存在时，脚本会停止安装，避免初始化新数据库。后续升级持续复用 `GEO_DATA_DIR`。

## Nginx

```nginx
server {
    listen 443 ssl http2;
    server_name geo.example.com;
    root /srv/geo-assessment/current/public;
    index index.php;

    location / {
        try_files $uri $uri/ /index.php?$query_string;
    }

    location ~ \.php$ {
        include fastcgi_params;
        fastcgi_param SCRIPT_FILENAME $document_root$fastcgi_script_name;
        fastcgi_param GEO_DATA_DIR /srv/geo-assessment/shared;
        fastcgi_param GEO_PUBLIC_URL https://geo.example.com;
        fastcgi_pass unix:/run/php/php7.3-fpm.sock;
    }

    location ~ /\. { deny all; }
}
```

部署到已有站点的子目录时，参考 [`../deploy/nginx-subdirectory.conf.example`](../deploy/nginx-subdirectory.conf.example)。示例中的 URL 前缀、项目绝对路径和 PHP-FPM socket 都需要替换。

## Apache

启用 `mod_rewrite`，将 DocumentRoot 指向 `public/`。仓库内 `public/.htaccess` 会把不存在的文件转发到入口，并拒绝点文件访问。

## 共享主机查询式路由

主机无法启用重写时，可通过 `index.php?r=attempts/...` 访问动态路由。静态资源仍由 `public/assets/` 直接提供。生产地址建议开启重写，以获得简洁 URL。

## 发布顺序

1. 上传到独立版本目录并安装生产依赖。
2. 配置固定的 `GEO_DATA_DIR`、`GEO_PUBLIC_URL` 和共享数据目录权限。
3. 对现有数据库执行 `backup:create` 与 `backup:verify`；首次采用共享目录时按上文设置 `GEO_LEGACY_DATA_DIR`。
4. 运行安装、题库验证、健康检查和本机回环冒烟测试。
5. 保持 `GEO_DATA_DIR` 不变，切换站点软链接或文档根。
6. 验证首页、答题保存、交卷、报告、打印和删除测试数据。

反向代理使用 HTTPS 时，设置 `GEO_TRUST_PROXY=1`，并限制只有受信代理可以写入 `X-Forwarded-Proto`。应用会在受信配置下根据 `X-Forwarded-Proto: https` 设置 Secure Cookie。

公共源码不启用访问统计。需要百度统计时，通过 PHP-FPM 环境或 Nginx `fastcgi_param` 设置 `GEO_BAIDU_ANALYTICS_ID`，并更新站点隐私说明。统计只在未登录首页启用，身份、答题、报告和证书页面始终关闭。官方演示配置位于 `../deploy/examples/ai.laoyao.cn/`。
