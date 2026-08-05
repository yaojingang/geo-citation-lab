# 部署说明

## 构建与首次安装

```bash
composer install --no-dev --classmap-authoritative
php bin/console app:install
php bin/console questions:verify
php bin/console app:health
```

站点文档根必须设为项目的 `public/`。将整个项目目录暴露为文档根会增加数据库和密钥泄露风险。

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
        fastcgi_pass unix:/run/php/php7.3-fpm.sock;
    }

    location ~ /\. { deny all; }
}
```

## Apache

启用 `mod_rewrite`，将 DocumentRoot 指向 `public/`。仓库内 `public/.htaccess` 会把不存在的文件转发到入口，并拒绝点文件访问。

## 共享主机查询式路由

主机无法启用重写时，可通过 `index.php?r=attempts/...` 访问动态路由。静态资源仍由 `public/assets/` 直接提供。生产地址建议开启重写，以获得简洁 URL。

## 发布顺序

1. 上传到独立版本目录并安装生产依赖。
2. 配置环境变量和 `storage/` 权限。
3. 对现有数据库执行 `backup:create` 与 `backup:verify`。
4. 运行安装、题库验证、健康检查和本机回环冒烟测试。
5. 切换站点软链接或文档根。
6. 验证首页、答题保存、交卷、报告、打印和删除测试数据。

反向代理使用 HTTPS 时，设置 `GEO_TRUST_PROXY=1`，并限制只有受信代理可以写入 `X-Forwarded-Proto`。应用会在受信配置下根据 `X-Forwarded-Proto: https` 设置 Secure Cookie。
