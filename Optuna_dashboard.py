from optuna_dashboard import run_server
import optuna
storage_url = "mysql://optuna:password@127.0.0.1:3306/optuna_db"
study = optuna.create_study(study_name="RandomSearch2", storage=storage_url, load_if_exists=True,
                            direction='minimize',
                            sampler=optuna.samplers.RandomSampler())  # erstellt studie und verbindet mit sql datenbank, erstellt objekt mit dem ich mit optuna studie interagieren kann

print(f"Best Trial so far: {study.best_trial.params}")
print(f"Best Penaltyso far: {study.best_value}")

run_server(storage_url)
