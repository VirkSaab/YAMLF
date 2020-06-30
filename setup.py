from distutils.core import setup
setup(
  name = 'yamlf',         # How you named your package folder (MyLib)
  packages = ['yamlf'],   # Chose the same as "name"
  version = '0.1.5',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Yet Another Machine Learning Framework (YAMLF)* is a lite machine learning and deep learning models training and inference framework. It contains data augmentations, data loaders, training routines that we usally rewrite for every ML project.',   # Give a short description about your library
  author = 'Jitender Singh Virk',                   # Type in your name
  author_email = 'krivsj@gmail.com',      # Type in your E-Mail
  url = 'https://github.com/VirkSaab/YAMLF',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/VirkSaab/YAMLF/archive/v0.1.5.tar.gz',    # release url
  keywords = ['Machine Learning', 'Deep Learning', 'PyTorch'],   # Keywords that define your package best
  install_requires=[            # dependencies
          'torch',
          'torchvision',
          'numpy',
          'albumentations',
          'pandas',
          'matplotlib',
          'scikit-learn',
          'fastprogress'
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
  ],
)
