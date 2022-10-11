# src

参考链接：
- [【实用工具】怎样快速访问Github？](https://blog.csdn.net/weixin_41512747/article/details/125941762?utm_medium=distribute.pc_feed_404.none-task-blog-2~default~BlogCommendFromBaidu~Rate-4-125941762-blog-null.pc_404_mixedpudn&depth_1-utm_source=distribute.pc_feed_404.none-task-blog-2~default~BlogCommendFromBaidu~Rate-4-125941762-blog-null.pc_404_mixedpud)
- [同一台电脑同时使用gitHub和gitLab](https://blog.csdn.net/m0_51691302/article/details/125706793)
- [怎么将本地文件上传到远程git仓库](https://www.cnblogs.com/wujindong/p/7280847.html)
- [GitHub如何删除项目库Repositories](https://www.likecs.com/show-203647457.html)
- [解决git@github.com: Permission denied (publickey)](https://blog.csdn.net/qq_40047019/article/details/122898308)

解决方案：
- 情形一：

如果ssh -T git@github.com出现：git@github.com: Permission denied (publickey).

则执行下述操作：

- ssh -v git@github.com

- ssh-agent -s

- ssh-add ~/.ssh/github_id_rsa

如果还是报错，则继续执行下述操作：
- eval `ssh-agent -s`
- ssh-add ~/.ssh/github_id_rsa
- ssh -T git@github.com
出现下述表述则表示正确：
Hi caoyuyu2021! You've successfully authenticated, but GitHub does not provide shell access.
