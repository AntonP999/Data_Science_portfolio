The solution script for "DOTA 2 win prediction" competition.

The script produces a submission file for 0.85360 public LB score result by running 
"python dota_solution.py".

The PATH_TO_DATA variable must point to folder with data files provided by hosts before competition:

 - train_features.csv
 - train_targets.csv
 - test_features.csv
 - train_matches.jsonl
 - test_matches.jsonl

Solution instructions:

Reads data from train_features.csv, train_targets.csv, test_features.csv.

Reads extended data from train_matches.jsonl, test_matches.jsonl, which includes:	

 - Objectives data:

		CHAT_MESSAGE_TOWER_KILL
		CHAT_MESSAGE_TOWER_DENY
		CHAT_MESSAGE_BARRACKS_KILL
		CHAT_MESSAGE_AEGIS
		CHAT_MESSAGE_AEGIS_STOLEN
		CHAT_MESSAGE_DENIED_AEGIS
		CHAT_MESSAGE_FIRSTBLOOD
		CHAT_MESSAGE_ROSHAN_KILL

 - Player data:

		ability_upgrades
		max_hero_hit
		purchase_log
		kills_log
		buyback_log
		runes_log
		actions
		pings
		killed
		item_uses
		ability_uses
		hero_hits
		damage
		damage_taken
		hero_inventory
		nearby_creep_death_count
		account_id_hash

 - Player gold time series. Applies linear regression and uses beta
   coefficient as indicator 	of player gold growth speed.

Uses one-hot encoder to vectorize hero ids.

Scales new gold feature.

Makes aggregates of player features by team. Calculates totals, means and standard deviations. Scales those features.

Uses non-aggregated extended player features which increased rocauc value on cross-validation:

 - ability_uses
 - kills_log
 - max_hero_hit
 - runes_log

One-hot encodes "lobby_type" and "game_mode" features.

Joins all those features as independend variables matrix. Depended variable is "radiant_win".

Searches for best LightGBM parameters using GridSearchCV (commented out in solution).

Trains LGB model with found parameters. Does this 5 times on 5 folds and averages predictions for test data.

Saves submission.
