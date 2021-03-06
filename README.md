# Readme

## RNN:
### How to run
python3 final_train.py $1 $2

python3 final_test.py $2 $3

$1: train.txt path

$2: test.txt path

$3: predict file path

### Environment
channels:
  - pytorch
  - defaults

dependencies:
  - blas=1.0=mkl
  - boto=2.49.0=py37_0
  - boto3=1.9.66=py37_0
  - botocore=1.12.67=py37_0
  - ca-certificates=2020.12.8=haa95532_0
  - cachetools=4.2.0=pyhd3eb1b0_0
  - certifi=2020.12.5=py37haa95532_0
  - cffi=1.14.4=py37hcd4344a_0
  - chardet=4.0.0=py37haa95532_1003
  - cryptography=3.3.1=py37hcd4344a_0
  - cudatoolkit=10.2.89=h74a9793_1
  - cycler=0.10.0=py37_0
  - docutils=0.16=py37_1
  - freetype=2.10.4=hd328e21_0
  - gensim=3.8.0=py37hf9181ef_0
  - google-api-core=1.22.2=py37h21ff451_0
  - google-auth=1.21.3=py_0
  - google-cloud-core=1.5.0=pyhd3eb1b0_0
  - google-cloud-storage=1.35.0=pyhd3eb1b0_0
  - google-crc32c=1.1.0=py37h2bbff1b_1
  - google-resumable-media=1.2.0=pyhd3eb1b0_1
  - googleapis-common-protos=1.52.0=py37h21ff451_0
  - icc_rt=2019.0.0=h0cc432a_1
  - icu=58.2=ha925a31_3
  - idna=2.10=py_0
  - intel-openmp=2020.2=254
  - jmespath=0.10.0=py_0
  - jpeg=9b=hb83a4c4_2
  - kiwisolver=1.3.0=py37hd77b12b_0
  - libcrc32c=1.1.1=ha925a31_2
  - libpng=1.6.37=h2a8f88b_0
  - libprotobuf=3.13.0.1=h200bbdf_0
  - libtiff=4.1.0=h56a325e_1
  - lz4-c=1.9.2=hf4a77e7_3
  - matplotlib=3.3.2=0
  - matplotlib-base=3.3.2=py37hba9282a_0
  - mkl=2020.2=256
  - mkl-service=2.3.0=py37h196d8e1_0
  - mkl_fft=1.2.0=py37h45dec08_0
  - mkl_random=1.1.1=py37h47e9c7a_0
  - multidict=5.1.0=py37h2bbff1b_2
  - ninja=1.10.2=py37h6d14046_0
  - numpy=1.19.2=py37hadc3359_0
  - numpy-base=1.19.2=py37ha3acd2a_0
  - olefile=0.46=py37_0
  - openssl=1.1.1i=h2bbff1b_0
  - pandas=1.1.5=py37hf11a4ad_0
  - pillow=8.0.1=py37h4fa10fc_0
  - pip=20.3.1=py37haa95532_0
  - protobuf=3.13.0.1=py37ha925a31_1
  - pyasn1=0.4.8=py_0
  - pyasn1-modules=0.2.8=py_0
  - pycparser=2.20=py_2
  - pyopenssl=20.0.1=pyhd3eb1b0_1
  - pyparsing=2.4.7=py_0
  - pyqt=5.9.2=py37h6538335_2
  - pysocks=1.7.1=py37_1
  - python=3.7.5=h8c8aaf0_0
  - python-dateutil=2.8.1=py_0
  - pytorch=1.6.0=py3.7_cuda102_cudnn7_0
  - pytz=2020.4=pyhd3eb1b0_0
  - qt=5.9.7=vc14h73c81de_0
  - requests=2.25.1=pyhd3eb1b0_0
  - rsa=4.6=py_0
  - s3transfer=0.1.13=py37_0
  - setuptools=51.0.0=py37haa95532_2
  - sip=4.19.8=py37h6538335_0
  - six=1.15.0=py37haa95532_0
  - smart_open=2.0.0=py_0
  - sqlite=3.33.0=h2a8f88b_0
  - tk=8.6.10=he774522_0
  - torchvision=0.7.0=py37_cu102
  - tornado=6.1=py37h2bbff1b_0
  - urllib3=1.24.3=py37_0
  - vc=14.2=h21ff451_1
  - vs2015_runtime=14.27.29016=h5e58377_2
  - wheel=0.36.1=pyhd3eb1b0_0
  - win_inet_pton=1.1.0=py37haa95532_0
  - wincertstore=0.2=py37_0
  - xz=5.2.5=h62dcd97_0
  - yarl=1.5.1=py37he774522_0
  - zlib=1.2.11=h62dcd97_4
  - zstd=1.4.5=h04227a9_0
  - pip:
    - joblib==0.17.0
    - scikit-learn==0.23.2
    - scipy==1.5.4
    - threadpoolctl==2.1.0

## Bert

### How to run
python3 TFHub.py $1 $2 $3 $4 $5

$1: train.csv path

$2: test.csv path

$3: sample_submission.csv path

$4: predict file path

$5: model directory path

### Environment

channels:
  - pytorch
  - huggingface
  - defaults

dependencies:
  - _tflow_select=2.3.0=gpu
  - absl-py=0.11.0=pyhd3eb1b0_1
  - argon2-cffi=20.1.0=py37he774522_1
  - astunparse=1.6.3=py_0
  - async_generator=1.10=py37h28b3542_0
  - attrs=20.3.0=pyhd3eb1b0_0
  - backcall=0.2.0=py_0
  - blas=1.0=mkl
  - bleach=3.2.1=py_0
  - blinker=1.4=py37_0
  - brotlipy=0.7.0=py37h2bbff1b_1003
  - ca-certificates=2020.12.8=haa95532_0
  - certifi=2020.12.5=py37haa95532_0
  - cffi=1.14.3=py37hcd4344a_2
  - click=7.1.2=py_0
  - colorama=0.4.4=py_0
  - cryptography=3.3.1=py37hcd4344a_0
  - cudatoolkit=10.2.89=h74a9793_1
  - dataclasses=0.7=py37_0
  - decorator=4.4.2=py_0
  - defusedxml=0.6.0=py_0
  - entrypoints=0.3=py37_0
  - filelock=3.0.12=py_0
  - freetype=2.10.4=hd328e21_0
  - google-auth-oauthlib=0.4.2=pyhd3eb1b0_2
  - google-pasta=0.2.0=py_0
  - h5py=2.10.0=py37h5e291fa_0
  - hdf5=1.10.4=h7ebc959_0
  - icc_rt=2019.0.0=h0cc432a_1
  - icu=58.2=ha925a31_3
  - idna=2.10=py_0
  - importlib_metadata=2.0.0=1
  - intel-openmp=2020.2=254
  - ipykernel=5.3.4=py37h5ca1d4c_0
  - ipython=7.19.0=py37hd4e2768_0
  - ipython_genutils=0.2.0=py37_0
  - ipywidgets=7.5.1=py_1
  - jedi=0.17.2=py37_0
  - jinja2=2.11.2=py_0
  - jpeg=9b=hb83a4c4_2
  - jsonschema=3.2.0=py_2
  - jupyter=1.0.0=py37_7
  - jupyter_client=6.1.7=py_0
  - jupyter_console=6.2.0=py_0
  - jupyter_core=4.7.0=py37haa95532_0
  - jupyterlab_pygments=0.1.2=py_0
  - keras-applications=1.0.8=py_1
  - libpng=1.6.37=h2a8f88b_0
  - libprotobuf=3.13.0.1=h200bbdf_0
  - libsodium=1.0.18=h62dcd97_0
  - libtiff=4.1.0=h56a325e_1
  - lz4-c=1.9.2=hf4a77e7_3
  - m2w64-gcc-libgfortran=5.3.0=6
  - m2w64-gcc-libs=5.3.0=7
  - m2w64-gcc-libs-core=5.3.0=7
  - m2w64-gmp=6.1.0=2
  - m2w64-libwinpthread-git=5.0.0.4634.697f757=2
  - markdown=3.3.3=py37haa95532_0
  - markupsafe=1.1.1=py37hfa6e2cd_1
  - mistune=0.8.4=py37hfa6e2cd_1001
  - mkl=2020.2=256
  - mkl-service=2.3.0=py37h2bbff1b_0
  - mkl_fft=1.2.0=py37h45dec08_0
  - mkl_random=1.1.1=py37h47e9c7a_0
  - msys2-conda-epoch=20160418=1
  - nbclient=0.5.1=py_0
  - nbconvert=6.0.7=py37_0
  - nbformat=5.0.8=py_0
  - nest-asyncio=1.4.3=pyhd3eb1b0_0
  - ninja=1.10.1=py37h7ef1ec2_0
  - notebook=6.1.4=py37_0
  - oauthlib=3.1.0=py_0
  - olefile=0.46=py37_0
  - openssl=1.1.1i=h2bbff1b_0
  - opt_einsum=3.1.0=py_0
  - packaging=20.4=py_0
  - pandoc=2.11=h9490d1a_0
  - pandocfilters=1.4.3=py37haa95532_1
  - parso=0.7.0=py_0
  - pickleshare=0.7.5=py37_1001
  - pillow=8.0.1=py37h4fa10fc_0
  - pip=20.2.4=py37haa95532_0
  - prometheus_client=0.8.0=py_0
  - prompt-toolkit=3.0.8=py_0
  - prompt_toolkit=3.0.8=0
  - pyasn1=0.4.8=py_0
  - pycparser=2.20=py_2
  - pygments=2.7.2=pyhd3eb1b0_0
  - pyjwt=1.7.1=py37_0
  - pyopenssl=20.0.1=pyhd3eb1b0_1
  - pyparsing=2.4.7=py_0
  - pyqt=5.9.2=py37h6538335_2
  - pyreadline=2.1=py37_1
  - pyrsistent=0.17.3=py37he774522_0
  - pysocks=1.7.1=py37_1
  - python=3.7.9=h60c2a47_0
  - python-dateutil=2.8.1=py_0
  - python_abi=3.7=1_cp37m
  - pytorch=1.6.0=py3.7_cuda102_cudnn7_0
  - pywin32=227=py37he774522_1
  - pywinpty=0.5.7=py37_0
  - pyzmq=19.0.2=py37ha925a31_1
  - qt=5.9.7=vc14h73c81de_0
  - qtconsole=4.7.7=py_0
  - qtpy=1.9.0=py_0
  - regex=2020.11.13=py37h2bbff1b_0
  - requests-oauthlib=1.3.0=py_0
  - rsa=4.6=py_0
  - send2trash=1.5.0=py37_0
  - setuptools=50.3.1=py37haa95532_1
  - sip=4.19.8=py37h6538335_0
  - six=1.15.0=py37haa95532_0
  - sqlite=3.33.0=h2a8f88b_0
  - tensorflow-base=2.3.0=eigen_py37h17acbac_0
  - terminado=0.9.1=py37_0
  - testpath=0.4.4=py_0
  - tk=8.6.10=he774522_0
  - torchvision=0.7.0=py37_cu102
  - tornado=6.0.4=py37he774522_1
  - traitlets=5.0.5=py_0
  - urllib3=1.26.2=pyhd3eb1b0_0
  - vc=14.2=h21ff451_1
  - vs2015_runtime=14.27.29016=h5e58377_2
  - wcwidth=0.2.5=py_0
  - webencodings=0.5.1=py37_1
  - werkzeug=1.0.1=py_0
  - wheel=0.35.1=pyhd3eb1b0_0
  - widgetsnbextension=3.5.1=py37_0
  - win_inet_pton=1.1.0=py37haa95532_0
  - wincertstore=0.2=py37_0
  - winpty=0.4.3=4
  - wrapt=1.12.1=py37he774522_1
  - xz=5.2.5=h62dcd97_0
  - zeromq=4.3.2=ha925a31_3
  - zipp=3.4.0=pyhd3eb1b0_0
  - zlib=1.2.11=h62dcd97_4
  - zstd=1.4.5=h04227a9_0
  - pip:
    - astor==0.8.1
    - bert-for-tf2==0.14.7
    - bert-tensorflow==1.0.1
    - cached-property==1.5.2
    - cachetools==4.1.1
    - chardet==3.0.4
    - cycler==0.10.0
    - cython==0.29.14
    - erlastic==2.0.0
    - gast==0.2.2
    - gensim==3.8.3
    - google-auth==1.23.0
    - grpcio==1.33.2
    - importlib-metadata==3.1.0
    - joblib==0.17.0
    - keras==2.4.0
    - keras-preprocessing==1.1.2
    - kiwisolver==1.3.1
    - matplotlib==3.3.2
    - nltk==3.5
    - numpy==1.18.5
    - opt-einsum==3.3.0
    - pandas==1.1.4
    - params-flow==0.8.2
    - protobuf==3.14.0
    - py-params==0.9.7
    - pyasn1-modules==0.2.8
    - pyspellchecker==0.5.5
    - pytz==2020.4
    - pyyaml==5.3.1
    - requests==2.25.0
    - sacremoses==0.0.43
    - scikit-learn==0.23.2
    - scipy==1.4.1
    - sentencepiece==0.1.94
    - smart-open==4.0.1
    - tensorboard==1.15.0
    - tensorboard-plugin-wit==1.7.0
    - tensorflow==2.1.0
    - tensorflow-estimator==1.15.1
    - tensorflow-gpu==1.15.0
    - tensorflow-hub==0.10.0
    - termcolor==1.1.0
    - threadpoolctl==2.1.0
    - tokenization==1.0.7
    - tokenizers==0.9.4
    - tqdm==4.50.2
