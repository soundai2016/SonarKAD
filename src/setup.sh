echo 'Start to build pypi...'
python3 -m pip install --upgrade build
python3 -m build
echo 'Finished build pypi...'
ls dist/