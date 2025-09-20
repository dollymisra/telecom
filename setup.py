from setuptools import setup, find_packages
setup(
    name='telecom_project',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scikit-learn',
        'joblib',
        'sqlalchemy'
    ],
    entry_points={
        'console_scripts': [
            'telecom-project=telecom_project_analysis:run_full_pipeline',
        ],
    },
)
