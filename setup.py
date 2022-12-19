from setuptools import setup,find_packages

def get_version(fp) -> str:
    with open(fp, "r") as f:
        for line in f:
            if "version" in line:
                delim = '"'
                return line.split(delim)[1]
    raise RuntimeError(f"could not find a valid version string in {fp}.")


setup(name='pattern_walker',
      version=get_version('pattern_walker/__init__.py'),
      description='A random walker model for browsing the law.',
      url='http://github.com/YPFoerster/pattern_walker',
      author='Yanik Foerster',
      author_email='ypfoerster@gmail.com',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
