由于这个机器无法连通外网，有一些 wheel 包无法安装，我已经放在了一个目录下面，首先执行指令

pip3 install --no-index --find-links=/mnt/public/daiyongxuan/wheels/verl \
    setuptools wheel

执行指令确认一下安装的结果：

python3 -c "import setuptools, wheel; print(setuptools.__version__, wheel.__version__)"

之后强制安装一下这两个包

PIP_NO_BUILD_ISOLATION=1 pip3 install -e . --no-deps --no-index \
  --find-links=/mnt/public/daiyongxuan/wheels/verl

环境应该就配置好了