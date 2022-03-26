from setuptools import setup

setup(name='backboned_unet',
      version='0.0.2',
      description='(Attention) U-Net built with TorchVision backbones.',
      url='https://github.com/erik-koynov/backboned-unet',
      keywords='machine deep learning neural networks pytorch torchvision segmentation unet',
      author='mate Kisantal. Erik Koynov',
      author_email='erik.koynov@gmail.com',
      license='MIT',
      packages=['backboned_unet'],
      install_requires=[
          'torch',
          'torchvision'
      ],
      zip_safe=False)
