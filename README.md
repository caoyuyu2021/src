# Resource

参考链接：
- [同一台电脑同时使用gitHub和gitLab](https://blog.csdn.net/m0_51691302/article/details/125706793)
- [GITHUB上传文件方法教程](https://blog.csdn.net/weixin_44161567/article/details/120064658)
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
Hi caoyuyu2021! You've successfully authenticated, but GitHub does not provide shell access.
