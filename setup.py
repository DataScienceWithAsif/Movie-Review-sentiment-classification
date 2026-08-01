import setuptools

__version__ = "0.0.0"

REPO_NAME = "Movie-Review-sentiment-classification"
Author_USER_NAME = "DataScienceWithAsif"
SRC_REPO = "FT_Model"
AUTHOR_EMAIL = "muasif025@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=Author_USER_NAME,
    author_email=AUTHOR_EMAIL,
    small_description="A small python package for movie review sentiment prediction",
    url=f"https://github.com/{Author_USER_NAME}/{REPO_NAME}",
    project_urls={
        "Bug Tracker": f"https://github.com/{Author_USER_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)