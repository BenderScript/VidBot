from streamlit.web import bootstrap

real_script = 'home.py'
bootstrap.run(real_script, f'run.py {real_script}', [], {})