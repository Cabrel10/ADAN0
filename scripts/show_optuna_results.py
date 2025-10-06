import optuna
import logging
import sys
import pandas as pd

def show_results():
    """
    Loads an Optuna study, displays a statistical summary of the top and most recent trials,
    and generates insightful visualization plots.
    """
    storage_name = "sqlite:///hyperparam_optimization.db"
    study_name_standard = "adan_trading_bot_hyperparam_optimization"
    study_name_ga = "adan_trading_bot_ga_optimization"
    study = None

    try:
        study = optuna.load_study(study_name=study_name_standard, storage=storage_name)
        logging.info(f"Successfully loaded study: '{study_name_standard}'")
    except (KeyError, ValueError):
        logging.warning(f"Study '{study_name_standard}' not found. Trying to load GA study '{study_name_ga}'.")
        try:
            study = optuna.load_study(study_name=study_name_ga, storage=storage_name)
            logging.info(f"Successfully loaded study: '{study_name_ga}'")
        except (KeyError, ValueError):
            logging.error(f"Neither study '{study_name_standard}' nor '{study_name_ga}' found in '{storage_name}'.")
            logging.info("You can list all available studies with the Optuna CLI: `optuna studies --storage sqlite:///hyperparam_optimization.db`")
            return

    all_trials = study.trials
    completed_trials = []
    pruned_trials = []
    failed_trials = []

    for t in all_trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            completed_trials.append(t)
        elif t.state == optuna.trial.TrialState.PRUNED:
            pruned_trials.append(t)
        elif t.state == optuna.trial.TrialState.FAIL:
            failed_trials.append(t)

    if not all_trials:
        logging.info("The study is empty. No trials to display.")
        return

    # --- Study Status Counter ---
    logging.info(f"\n--- STATISTICAL ANALYSIS OF STUDY '{study.study_name}' ---")
    logging.info(f"Total trials registered: {len(all_trials)}")
    logging.info(f"  - ✅ Completed: {len(completed_trials)}")
    logging.info(f"  -  pruned: {len(pruned_trials)}")
    logging.info(f"  - ❌ Failed:    {len(failed_trials)}")

    # Sort completed trials by score for ranking
    completed_trials.sort(key=lambda t: t.value, reverse=True)

    # --- Display Most Recent Trial ---
    if completed_trials:
        last_completed_trial = sorted(completed_trials, key=lambda t: t.number)[-1]
        logging.info("\n--- MOST RECENT COMPLETED TRIAL ---")
        print(f"\n🕒 LAST | Trial #{last_completed_trial.number}")
        print(f"   Value (Sharpe Ratio): {last_completed_trial.value:.5f}")
        print("   Parameters:")
        for key, value in last_completed_trial.params.items():
            if isinstance(value, float):
                print(f"     - {key}: {value:.6f}")
            else:
                print(f"     - {key}: {value}")
        print("   User Attributes:")
        for key, value in last_completed_trial.user_attrs.items():
            print(f"     - {key}: {value}")

    # --- Display Top 3 Trials ---
    if completed_trials:
        logging.info("\n--- TOP 3 BEST TRIALS ---")
        for i, trial in enumerate(completed_trials[:3]):
            print(f"\n🏆 RANK {i+1} | Trial #{trial.number}")
            print(f"   Value (Sharpe Ratio): {trial.value:.5f}")
            print("   Parameters:")
            for key, value in trial.params.items():
                if isinstance(value, float):
                    print(f"     - {key}: {value:.6f}")
                else:
                    print(f"     - {key}: {value}")
            print("   User Attributes:")
            for key, value in trial.user_attrs.items():
                print(f"     - {key}: {value}")
    else:
        logging.info("\nNo completed trials yet to rank.")

    # --- Generate and save visualization plots ---
    logging.info("\n--- PLOT GENERATION ---")
    try:
        import optuna.visualization as vis

        # 1. Optimization History Plot
        fig_history = vis.plot_optimization_history(study)
        fig_history.write_html("optuna_history.html")
        logging.info("✅ Generated optimization history plot: optuna_history.html")

        # 2. Slice Plot
        fig_slice = vis.plot_slice(study)
        fig_slice.write_html("optuna_slice.html")
        logging.info("✅ Generated slice plot: optuna_slice.html")

        # 3. Hyperparameter Importances Plot
        try:
            fig_importance = vis.plot_param_importances(study)
            fig_importance.write_html("optuna_importance.html")
            logging.info("✅ Generated parameter importance plot: optuna_importance.html")
        except Exception as e:
            logging.warning(f"⚠️ Could not generate parameter importance plot. This is common in early stages if all trials have similar scores. Error: {e}")

    except ImportError:
        logging.warning("\nCould not generate plots. Please install plotly and kaleido: `pip install plotly kaleido`")
    except Exception as e:
        logging.error(f"\nAn unexpected error occurred during plot generation: {e}")

if __name__ == "__main__":
    show_results()
