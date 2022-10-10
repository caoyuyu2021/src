# Resource

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

参考链接：
[同一台电脑同时使用gitHub和gitLab](https://blog.csdn.net/m0_51691302/article/details/125706793)
