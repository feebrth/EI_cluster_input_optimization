from optuna_dashboard import run_server

storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"



run_server(storage_url)
