from pathlib import Path
import sys
import unittest


REPO_ROOT = Path(__file__).resolve().parents[1]
CUIHU_DIR = REPO_ROOT / "scripts" / "cuihu"
CUIHU_PATH = str(CUIHU_DIR)
if CUIHU_PATH in sys.path:
    sys.path.remove(CUIHU_PATH)
sys.path.insert(0, CUIHU_PATH)


class RFPOAutoIterateTests(unittest.TestCase):
    def test_monitor_ranks_only_post_training_eval_points(self):
        from monitor_rfpo_scratch import EvalPoint, trained_points

        root = Path("/tmp/rfpo")
        points = [
            EvalPoint(
                run_name="rfpo_scratch_best75_start_eval_noise",
                step=0,
                reward=75.3,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.0045,
                source_log=root / "logs" / "run.out",
            ),
            EvalPoint(
                run_name="rfpo_scratch_best74_femto_lr10em7_std0022_clip002_80m",
                step=18_796_544,
                reward=75.2,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.00429,
                source_log=root / "logs" / "run.out",
            ),
        ]

        ranked = trained_points(points)

        self.assertEqual([point.run_name for point in ranked], [points[1].run_name])

    def test_best_point_prefers_precision_when_rewards_are_close(self):
        from monitor_rfpo_scratch import EvalPoint, best_point

        root = Path("/tmp/rfpo")
        points = [
            EvalPoint(
                run_name="rfpo_scratch_reward_high_precision_bad",
                step=21_196_800,
                reward=75.6,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.00436,
                source_log=root / "logs" / "bad.out",
            ),
            EvalPoint(
                run_name="rfpo_scratch_reward_close_precision_good",
                step=21_598_208,
                reward=75.2,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.00292,
                source_log=root / "logs" / "good.out",
            ),
        ]

        self.assertEqual(best_point(points).run_name, "rfpo_scratch_reward_close_precision_good")

    def test_best_point_still_rejects_unsolved_stage(self):
        from monitor_rfpo_scratch import EvalPoint, best_point

        root = Path("/tmp/rfpo")
        points = [
            EvalPoint(
                run_name="rfpo_scratch_unsolved_but_precise",
                step=10_000_000,
                reward=76.0,
                stage=1.0,
                pregrasp=1.0,
                obj_err=0.001,
                source_log=root / "logs" / "bad.out",
            ),
            EvalPoint(
                run_name="rfpo_scratch_solved_stage",
                step=10_000_000,
                reward=75.0,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.004,
                source_log=root / "logs" / "good.out",
            ),
        ]

        self.assertEqual(best_point(points).run_name, "rfpo_scratch_solved_stage")

    def test_refinement_candidate_resumes_from_best_checkpoint(self):
        from auto_iterate_rfpo_scratch import refinement_candidate
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        best = EvalPoint(
            run_name="rfpo_scratch_best72_micro_lr05_std035_clip04_80m",
            step=13_799_424,
            reward=73.6,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00476,
            source_log=root / "logs" / "run.out",
        )

        candidate = refinement_candidate(root, best, gpu=2, generation=0)

        self.assertEqual(candidate.gpu, 2)
        self.assertIn("best73", candidate.run)
        self.assertIn("s13799424", candidate.run)
        self.assertIn(
            "resume_model=/tmp/rfpo/results/vividex_fpo_runtime/results/state_baseline/"
            "rfpo_scratch_best72_micro_lr05_std035_clip04_80m/models/best.pt",
            candidate.overrides,
        )
        self.assertIn("agent.params.actor_objective=gaussian_ppo", candidate.overrides)
        self.assertIn("agent.params.trust_region_mode=ppo", candidate.overrides)
        self.assertIn("agent.params.cfm_loss_reduction=mean", candidate.overrides)
        self.assertIn("agent.params.cfm_loss_use_huber=True", candidate.overrides)
        self.assertIn("agent.params.cfm_loss_t_inverse_cdf_beta=1.5", candidate.overrides)
        self.assertIn("agent.params.cfm_diff_clip_max=null", candidate.overrides)
        self.assertIn("agent.params.fpo_objective_coef=0.0", candidate.overrides)
        self.assertIn("agent.params.lr_schedule=fixed", candidate.overrides)
        self.assertIn("agent.params.target_kl=0.00055", candidate.overrides)
        self.assertIn("agent.params.max_clip_fraction=0.07", candidate.overrides)
        self.assertIn("agent.params.residual_action_scale=0.005", candidate.overrides)
        self.assertIn("agent.params.gaussian_action_std=0.008", candidate.overrides)
        self.assertIn("agent.params.on_policy_action_anchor_coef=0.090", candidate.overrides)
        self.assertIn("agent.params.on_policy_action_anchor_min_coef=0.090", candidate.overrides)
        self.assertIn("agent.params.obj_precision_reward_coef=4.0", candidate.overrides)
        self.assertIn("agent.params.obj_precision_reward_stage_min=2.0", candidate.overrides)
        self.assertIn("agent.params.precision_elite_action_coef=0.045", candidate.overrides)
        self.assertIn("agent.params.precision_elite_err_threshold=0.0020", candidate.overrides)
        self.assertIn("agent.params.precision_elite_min_fraction=0.02", candidate.overrides)
        self.assertIn("agent.params.precision_elite_mode=gaussian_nll", candidate.overrides)
        self.assertIn("env.task_kwargs.reward_kwargs.obj_err_scale=50", candidate.overrides)
        self.assertIn("env.task_kwargs.reward_kwargs.object_reward_scale=10.0", candidate.overrides)
        self.assertIn("agent.params.obj_precision_penalty_coef=1250.0", candidate.overrides)
        self.assertIn("agent.params.obj_precision_penalty_target=0.0011", candidate.overrides)
        self.assertIn("agent.params.obj_precision_penalty_power=1.0", candidate.overrides)
        self.assertIn("agent.params.obj_precision_penalty_max=2.8", candidate.overrides)
        self.assertIn("agent.params.obj_precision_delta_coef=360.0", candidate.overrides)
        self.assertIn("agent.params.obj_precision_delta_target=0.0011", candidate.overrides)
        self.assertIn("agent.params.obj_precision_delta_max=1.3", candidate.overrides)

    def test_refinement_scheduler_cycles_precision_candidates(self):
        from auto_iterate_rfpo_scratch import REFINEMENT_GRID, RFPO_REFINEMENT_GENERATIONS

        scheduled = REFINEMENT_GRID[:RFPO_REFINEMENT_GENERATIONS]

        self.assertEqual(
            [cfg["tag"] for cfg in scheduled],
            [
                "prec_v2_lr30_std008_anchor090",
                "prec_v2_lr35_std009_anchor085",
                "prec_v2_lr40_std010_anchor080",
                "prec_v2_lr25_std008_anchor100",
                "prec_v2_lr45_std011_anchor075",
                "prec_v2_action_lr35_std008_anchor090",
                "prec_v2_lr30_std007_anchor105",
                "prec_v2_hybrid_lr35_std009_anchor085",
            ],
        )
        self.assertTrue(all(cfg["objective"] in {"gaussian_ppo", "hybrid_fpo"} for cfg in scheduled))
        self.assertGreaterEqual(
            sum(1 for cfg in scheduled if cfg["objective"] == "gaussian_ppo"),
            6,
        )
        self.assertTrue(any(cfg["objective"] == "hybrid_fpo" for cfg in scheduled))
        self.assertTrue(all(0.004 <= float(cfg["residual"]) <= 0.008 for cfg in scheduled))
        self.assertTrue(all(0.007 <= float(cfg["std"]) <= 0.011 for cfg in scheduled))
        self.assertTrue(all(cfg["env_obj_err_scale"] == "50" for cfg in scheduled))
        self.assertTrue(all(cfg["env_object_reward_scale"] == "10.0" for cfg in scheduled))
        self.assertTrue(all(float(cfg["on_policy_anchor"]) > 0.0 for cfg in scheduled))
        self.assertTrue(all(0.075 <= float(cfg["on_policy_anchor"]) <= 0.105 for cfg in scheduled))
        self.assertTrue(all(float(cfg["on_policy_anchor_min"]) == float(cfg["on_policy_anchor"]) for cfg in scheduled))
        self.assertTrue(all(int(cfg["on_policy_anchor_decay"]) == 0 for cfg in scheduled))
        self.assertTrue(all(float(cfg["obj_precision_coef"]) > 0.0 for cfg in scheduled))
        self.assertTrue(all(float(cfg["obj_precision_penalty_coef"]) > 0.0 for cfg in scheduled))
        self.assertTrue(all(float(cfg.get("obj_precision_delta_coef", "0.0")) > 0.0 for cfg in scheduled))
        self.assertTrue(all(float(cfg["awr_coef"]) > 0.0 for cfg in scheduled))
        self.assertTrue(all(cfg["awr_positive_only"] == "True" for cfg in scheduled))
        self.assertTrue(all(2.5e-7 <= float(cfg["lr"]) <= 4.5e-7 for cfg in scheduled))
        self.assertGreaterEqual(
            sum(float(cfg.get("precision_elite_coef", "0.0")) > 0.0 for cfg in scheduled),
            3,
        )

    def test_refinement_candidate_has_tiny_hybrid_variant(self):
        from auto_iterate_rfpo_scratch import refinement_candidate, override_value
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        best = EvalPoint(
            run_name="rfpo_scratch_best73_nano_lr03_std03_clip03_80m",
            step=14_000_000,
            reward=74.2,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.0045,
            source_log=root / "logs" / "run.out",
        )

        steady = refinement_candidate(root, best, gpu=1, generation=0)
        later = refinement_candidate(root, best, gpu=1, generation=4)

        self.assertEqual(override_value(steady.overrides, "agent.params.fpo_objective_coef"), "0.0")
        self.assertEqual(override_value(later.overrides, "agent.params.fpo_objective_coef"), "0.0")
        self.assertEqual(override_value(later.overrides, "agent.params.actor_objective"), "gaussian_ppo")
        self.assertEqual(override_value(later.overrides, "agent.params.action_head_mode"), "flow_residual")
        self.assertEqual(override_value(later.overrides, "agent.params.log_ratio_scale"), "0.0")

    def test_refinement_generation_usage_keeps_seen_generations_when_pool_is_exhausted(self):
        from auto_iterate_rfpo_scratch import REFINEMENT_GRID, refinement_generations_in_use
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        best = EvalPoint(
            run_name="rfpo_scratch_best75_elite_joint_c",
            step=21_598_208,
            reward=75.2,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00292,
            source_log=root / "logs" / "best.out",
        )
        points = [
            EvalPoint(
                run_name=(
                    f"rfpo_scratch_best75_{cfg['tag']}_rankelite2_s21598208_"
                    "lr10em6_std0018_clip004_80m"
                ),
                step=21_798_912,
                reward=74.9,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.004,
                source_log=root / "logs" / "seen.out",
            )
            for cfg in REFINEMENT_GRID[:8]
        ]

        used = refinement_generations_in_use(root, best, running=set(), points=points)

        self.assertEqual(used, set(range(8)))

    def test_refinement_stop_uses_precision_aware_score(self):
        from auto_iterate_rfpo_scratch import should_stop_refinement
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        global_best = EvalPoint(
            run_name="rfpo_scratch_best75_precision_good",
            step=21_598_208,
            reward=75.2,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00292,
            source_log=root / "logs" / "best.out",
        )
        lower_reward_better_precision = [
            EvalPoint(
                run_name="rfpo_scratch_candidate_precise",
                step=21_998_208,
                reward=74.8,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.0012,
                source_log=root / "logs" / "candidate.out",
            ),
            EvalPoint(
                run_name="rfpo_scratch_candidate_precise",
                step=22_398_208,
                reward=74.7,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.00115,
                source_log=root / "logs" / "candidate.out",
            ),
            EvalPoint(
                run_name="rfpo_scratch_candidate_precise",
                step=22_798_208,
                reward=74.9,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.00112,
                source_log=root / "logs" / "candidate.out",
            ),
        ]

        self.assertFalse(should_stop_refinement(lower_reward_better_precision, global_best))

    def test_base_overrides_do_not_force_direct_mlp_for_refinements(self):
        from auto_iterate_rfpo_scratch import BASE_OVERRIDES

        self.assertNotIn("agent.params.action_head_mode=direct_mlp", BASE_OVERRIDES)

    def test_first_refinement_uses_conservative_flow_policy_ppo_update(self):
        from auto_iterate_rfpo_scratch import refinement_candidate, override_value
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        best = EvalPoint(
            run_name="rfpo_scratch_best73_nano_lr03_std03_clip03_80m",
            step=14_000_000,
            reward=73.9,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.0045,
            source_log=root / "logs" / "run.out",
        )

        candidate = refinement_candidate(root, best, gpu=0, generation=0)

        self.assertIn("prec_v2_lr30_std008_anchor090", candidate.run)
        self.assertIn("agent.params.actor_objective=gaussian_ppo", candidate.overrides)
        self.assertIn("agent.params.gaussian_objective_coef=1.0", candidate.overrides)
        self.assertIn("agent.params.gaussian_objective_min_coef=1.0", candidate.overrides)
        self.assertIn("agent.params.fpo_objective_min_coef=0.0", candidate.overrides)
        self.assertIn("agent.params.log_ratio_scale=0.0", candidate.overrides)
        self.assertIn("agent.params.lr_schedule=fixed", candidate.overrides)
        self.assertIn("agent.params.target_kl=0.00055", candidate.overrides)
        self.assertEqual(override_value(candidate.overrides, "agent.params.residual_action_scale"), "0.005")
        self.assertEqual(override_value(candidate.overrides, "agent.params.on_policy_action_anchor_coef"), "0.090")
        self.assertEqual(override_value(candidate.overrides, "agent.params.on_policy_action_anchor_min_coef"), "0.090")
        self.assertEqual(override_value(candidate.overrides, "agent.params.on_policy_action_anchor_decay_steps"), "0")
        self.assertEqual(override_value(candidate.overrides, "agent.params.advantage_weighted_action_coef"), "0.00035")
        self.assertEqual(override_value(candidate.overrides, "agent.params.advantage_weighted_action_positive_only"), "True")
        self.assertEqual(override_value(candidate.overrides, "agent.params.advantage_weighted_action_max_weight"), "1.8")
        self.assertEqual(override_value(candidate.overrides, "agent.params.precision_elite_action_coef"), "0.045")
        self.assertEqual(override_value(candidate.overrides, "agent.params.precision_elite_mode"), "gaussian_nll")
        self.assertEqual(override_value(candidate.overrides, "agent.params.precision_elite_top_fraction"), "0.04")
        self.assertEqual(override_value(candidate.overrides, "agent.params.precision_elite_min_fraction"), "0.02")
        self.assertEqual(override_value(candidate.overrides, "agent.params.precision_elite_reward_quantile"), "0.88")
        self.assertEqual(override_value(candidate.overrides, "agent.params.precision_elite_hand_threshold"), "0.0190")
        self.assertEqual(override_value(candidate.overrides, "agent.params.precision_elite_control_threshold"), "0.0040")
        self.assertEqual(override_value(candidate.overrides, "env.task_kwargs.reward_kwargs.obj_err_scale"), "50")
        self.assertEqual(override_value(candidate.overrides, "env.task_kwargs.reward_kwargs.object_reward_scale"), "10.0")

    def test_second_refinement_uses_wider_flow_policy_ppo_candidate(self):
        from auto_iterate_rfpo_scratch import refinement_candidate
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        best = EvalPoint(
            run_name="rfpo_scratch_best75_hybridrfpo_lr20em7_std0024_clip0025_80m",
            step=17_000_000,
            reward=75.1,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.0039,
            source_log=root / "logs" / "run.out",
        )

        candidate = refinement_candidate(root, best, gpu=1, generation=1)

        self.assertIn("prec_v2_lr35_std009_anchor085", candidate.run)
        self.assertIn("agent.params.actor_objective=gaussian_ppo", candidate.overrides)
        self.assertIn("agent.params.trust_region_mode=ppo", candidate.overrides)
        self.assertIn("agent.params.cfm_loss_reduction=mean", candidate.overrides)
        self.assertIn("agent.params.fpo_objective_coef=0.0", candidate.overrides)
        self.assertIn("agent.params.lr_schedule=fixed", candidate.overrides)
        self.assertIn("agent.params.min_learning_rate=3.5e-7", candidate.overrides)
        self.assertIn("agent.params.max_learning_rate=3.5e-7", candidate.overrides)
        self.assertIn("agent.params.rollout_action_noise_std=0.0", candidate.overrides)
        self.assertIn("agent.params.action_perturb_std=0.0", candidate.overrides)
        self.assertIn("agent.params.advantage_weighted_action_coef=0.00045", candidate.overrides)
        self.assertIn("agent.params.precision_elite_action_coef=0.042", candidate.overrides)
        self.assertIn("agent.params.precision_elite_mode=gaussian_nll", candidate.overrides)
        self.assertIn("env.task_kwargs.reward_kwargs.object_reward_scale=10.0", candidate.overrides)

    def test_later_refinement_uses_object_focus_flowppo_candidate(self):
        from auto_iterate_rfpo_scratch import refinement_candidate
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        best = EvalPoint(
            run_name="rfpo_scratch_best74_femto_lr10em7_std0022_clip002_80m",
            step=18_796_544,
            reward=75.2,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00429,
            source_log=root / "logs" / "run.out",
        )

        candidate = refinement_candidate(root, best, gpu=2, generation=5)

        self.assertIn("prec_v2_action_lr35_std008_anchor090", candidate.run)
        self.assertIn("agent.params.actor_objective=gaussian_ppo", candidate.overrides)
        self.assertIn("agent.params.log_ratio_scale=0.0", candidate.overrides)
        self.assertIn("agent.params.cfm_diff_clip_max=null", candidate.overrides)
        self.assertIn("agent.params.advantage_clamp_positive=32.0", candidate.overrides)
        self.assertIn("agent.params.max_clip_fraction=0.075", candidate.overrides)
        self.assertIn("agent.params.advantage_weighted_action_coef=0.00035", candidate.overrides)
        self.assertIn("agent.params.advantage_weighted_action_mode=action", candidate.overrides)

    def test_refinement_generation_usage_is_scoped_to_current_best_source(self):
        from auto_iterate_rfpo_scratch import refinement_generations_in_use
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        best = EvalPoint(
            run_name="rfpo_scratch_best74_femto_lr10em7_std0022_clip002_80m",
            step=18_796_544,
            reward=75.2,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00429,
            source_log=root / "logs" / "run.out",
        )
        points = [
            EvalPoint(
                run_name=(
                    "rfpo_scratch_best75_prec_v2_lr45_std011_anchor075_"
                    "rankelite2_s18796544_lr45em7_std0011_clip0022_80m"
                ),
                step=18_997_248,
                reward=75.1,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.00486,
                source_log=root / "logs" / "run.out",
            ),
            EvalPoint(
                run_name=(
                    "rfpo_scratch_best75_prec_v2_hybrid_lr35_std009_anchor085_"
                    "rankelite2_s18796544_lr35em7_std0009_clip0018_80m"
                ),
                step=18_997_248,
                reward=75.1,
                stage=2.0,
                pregrasp=1.0,
                obj_err=0.00486,
                source_log=root / "logs" / "run.out",
            )
        ]
        running = {"rfpo_scratch_best75_prec_v2_lr30_std008_anchor090_rankelite2_s00000000_lr30em7_std0008_clip0016_80m"}

        used = refinement_generations_in_use(root, best, running=running, points=points)

        self.assertEqual(used, {0, 4, 7})

    def test_refinement_source_extracts_resume_run_name(self):
        from auto_iterate_rfpo_scratch import refinement_candidate, refinement_source
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        best = EvalPoint(
            run_name="rfpo_scratch_best72_micro_lr05_std035_clip04_80m",
            step=13_799_424,
            reward=73.6,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00476,
            source_log=root / "logs" / "run.out",
        )

        candidate = refinement_candidate(root, best, gpu=2, generation=0)

        self.assertEqual(refinement_source(candidate.overrides), best.run_name)

    def test_tick_does_not_launch_new_candidate_after_stopping_a_run(self):
        import unittest.mock as mock

        import auto_iterate_rfpo_scratch
        from monitor_rfpo_scratch import EvalPoint

        root = Path("/tmp/rfpo")
        repo = root / "repo"
        best = EvalPoint(
            run_name="rfpo_scratch_best75_elite_nll_clean_a_rankelite2_s28798976_lr50em7_std0014_clip0025_80m",
            step=28_798_976,
            reward=76.1,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00274,
            source_log=root / "logs" / "best.out",
        )
        weak = EvalPoint(
            run_name="rfpo_scratch_best76_elite_nll_lr50_std014_anchor055_rankelite2_s28798976_lr50em7_std0014_clip0025_80m",
            step=29_198_976,
            reward=75.1,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00310,
            source_log=root / "logs" / "weak.out",
        )
        weak_later = EvalPoint(
            run_name=weak.run_name,
            step=29_398_976,
            reward=75.0,
            stage=2.0,
            pregrasp=1.0,
            obj_err=0.00320,
            source_log=root / "logs" / "weak.out",
        )

        with mock.patch.object(auto_iterate_rfpo_scratch, "all_points", return_value=[best, weak, weak_later]), \
            mock.patch.object(auto_iterate_rfpo_scratch, "write_record"), \
            mock.patch.object(auto_iterate_rfpo_scratch, "python_processes", return_value={weak.run_name: (12345, "python tools/train.py")}), \
            mock.patch.object(auto_iterate_rfpo_scratch, "stop_run") as stop_run, \
            mock.patch.object(auto_iterate_rfpo_scratch, "launch") as launch, \
            mock.patch.object(auto_iterate_rfpo_scratch, "sync_desktop"):
            auto_iterate_rfpo_scratch.tick(root, repo)

        stop_run.assert_called_once_with(12345)
        launch.assert_not_called()

    def test_busy_gpus_counts_high_memory_but_ignores_small_cuda_contexts(self):
        import unittest.mock as mock

        import auto_iterate_rfpo_scratch

        with mock.patch.object(auto_iterate_rfpo_scratch, "sh") as fake_sh:
            fake_sh.return_value.stdout = "0, 68629\n1, 1447\n2, 1447\n3, 9395\n"

            busy = auto_iterate_rfpo_scratch.busy_gpus()

        self.assertEqual(busy, {0, 3})

    def test_busy_gpus_counts_rfpo_cuda_visible_devices(self):
        import unittest.mock as mock

        import auto_iterate_rfpo_scratch

        with mock.patch.object(auto_iterate_rfpo_scratch, "sh") as fake_sh:
            fake_sh.return_value.stdout = "0, 1447\n1, 1447\n2, 1447\n3, 1447\n"
            with mock.patch.object(
                auto_iterate_rfpo_scratch,
                "python_processes",
                return_value={"rfpo_scratch_test": (12345, "python tools/train.py")},
            ):
                with mock.patch.object(auto_iterate_rfpo_scratch, "process_cuda_visible_devices", return_value={2}):
                    busy = auto_iterate_rfpo_scratch.busy_gpus()

        self.assertEqual(busy, {2})

    def test_trainstd_counts_as_refinement_run(self):
        from auto_iterate_rfpo_scratch import is_refinement_run

        self.assertTrue(is_refinement_run("rfpo_scratch_best74_trainstd_lr50em7_std008_clip005_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best74_awr_lr50em7_std0035_clip004_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best74_flowawr_lr40em7_std0032_clip0035_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_fpo2sqrt_lr20em6_std006_clip005_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_purefpo2sqrt_lr10em6_std004_clip005_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_hybridsteady_lr10em7_std0022_clip002_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_fpoplusplus_lr80em8_std0022_clip002_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_hybridfpo1_lr15em7_std0022_clip002_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_fpoexplore_lr10em7_std0024_clip0025_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_hybridmicro_lr60em8_std0018_clip0015_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_hybridpos_lr80em8_std0018_clip0015_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_hybridanchor_lr10em7_std0020_clip0015_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_fpomicro_lr40em8_std0018_clip0015_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_hybridrfpo_lr20em7_std0024_clip0025_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_hybridrfpo_low_lr10em7_std0022_clip002_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_execppo_lr40em7_std0032_clip0035_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_flowppo_precise_lr30em6_std008_clip012_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_flowppo_wide_lr50em6_std012_clip015_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_hybridtiny_lr30em6_std008_clip012_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_guardppo_a_lr10em7_std0018_clip002_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_guardhybrid_a_lr20em7_std0022_clip0025_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_awrguard_a_lr20em7_std0022_clip0025_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_awrhybrid_a_lr20em7_std0022_clip0025_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_precision_residual_a_lr12em6_std0014_clip0035_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_precision_residual_b_lr80em7_std0012_clip003_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_precision_residual_c_lr15em6_std0016_clip004_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_precision_flow_a_lr80em7_std0018_clip004_80m"))
        self.assertTrue(is_refinement_run("rfpo_scratch_best75_weak_anchor_hybrid_a_rankelite2_s18796544_lr14em6_std0030_clip005_80m"))


if __name__ == "__main__":
    unittest.main()
