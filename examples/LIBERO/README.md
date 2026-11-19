# LIBERO data for Motion-DynVLA

The source policy trains on 30 tasks across LIBERO Spatial, Object and Goal. The target policy adapts to LIBERO Long using 20 demonstrations per task. The 500 evaluation episodes are simulator rollouts from 50 fixed initial states per target task.

## Training data

The launchers expect the converted LeRobot datasets under `$MOTION_DYNVLA_ROOT/data/libero`. Each directory keeps its own `meta`, `data` and `videos` files.

| Stage | Dataset directory |
| --- | --- |
| Source | `libero_spatial_no_noops_1.0.0_lerobot` |
| Source | `libero_object_no_noops_1.0.0_lerobot` |
| Source | `libero_goal_no_noops_1.0.0_lerobot` |
| Target | `libero_10_no_noops_1.0.0_lerobot` |

The [source manifest](../../configs/data_manifests/libero_source30.json) excludes episode 82 of LIBERO Goal. The [target manifest](../../configs/data_manifests/libero_target10_20shot.json) lists the 200 demonstrations used for adaptation. Set each `dataset_path` in these two files to the absolute path of the matching local dataset and keep the episode selections intact.

The [LIBERO training configuration](dynvla_libero_config.py) uses the camera keys `primary_image` and `wrist_image`. Check these names against the video entries in each converted dataset's `meta/modality.json`.

See [Training](../../README.md#training) for the three training stages and [Evaluation](../../README.md#evaluation) to evaluate the released target policy.

## Task list

### LIBERO 10 (Long)
- `libero_sim/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket`
- `libero_sim/LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket`
- `libero_sim/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it`
- `libero_sim/KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it`
- `libero_sim/LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate`
- `libero_sim/STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_back_compartment_of_the_caddy`
- `libero_sim/LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate`
- `libero_sim/LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket`
- `libero_sim/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove`
- `libero_sim/KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it`

### LIBERO Goal
- `libero_sim/open_the_middle_drawer_of_the_cabinet`
- `libero_sim/put_the_bowl_on_the_stove`
- `libero_sim/put_the_wine_bottle_on_top_of_the_cabinet`
- `libero_sim/open_the_top_drawer_and_put_the_bowl_inside`
- `libero_sim/put_the_bowl_on_top_of_the_cabinet`
- `libero_sim/push_the_plate_to_the_front_of_the_stove`
- `libero_sim/put_the_cream_cheese_in_the_bowl`
- `libero_sim/turn_on_the_stove`
- `libero_sim/put_the_bowl_on_the_plate`
- `libero_sim/put_the_wine_bottle_on_the_rack`

### LIBERO Object
- `libero_sim/pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_cream_cheese_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_salad_dressing_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_bbq_sauce_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_ketchup_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_tomato_sauce_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_butter_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_milk_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_chocolate_pudding_and_place_it_in_the_basket`
- `libero_sim/pick_up_the_orange_juice_and_place_it_in_the_basket`

### LIBERO Spatial
- `libero_sim/pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_cookie_box_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_in_the_top_drawer_of_the_wooden_cabinet_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_stove_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_next_to_the_plate_and_place_it_on_the_plate`
- `libero_sim/pick_up_the_black_bowl_on_the_wooden_cabinet_and_place_it_on_the_plate`
