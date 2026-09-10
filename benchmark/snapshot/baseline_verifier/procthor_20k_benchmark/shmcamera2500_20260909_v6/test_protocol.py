"""Configuration/resume/statistics regression; no GPU or simulator."""
import json
from pathlib import Path
import tempfile
import unittest
from protocol import CONFIG, METHODS, prepare_root, verify_root, stamp, digest
from resume_support import completed_or_archive
from summary import summarize
from execution_policy import policy_for


class ProtocolTests(unittest.TestCase):
    def row(self, task, success=False):
        return dict(**stamp(), method='our_model', task_sha256=digest(task),
            executor_policy=policy_for('our_model'),
            execution_mode=CONFIG['execution_mode'], robot=CONFIG['robot'],
            success=success, status='success' if success else 'collision',
            total_wall_seconds=5., resource_wall_seconds=6.,
            raw_executed_path_ratio=1.1 if success else None, spl=.9 if success else 0.)

    def test_counts_and_parameters(self):
        self.assertEqual(sum(CONFIG['sampling']['groups_per_source'].values()),500)
        self.assertEqual(CONFIG['sampling']['routes_per_group'],5)
        self.assertEqual(len(METHODS),7)
        self.assertEqual(CONFIG['robot']['yaw_rate_deg_s'],90)
        self.assertFalse(CONFIG['sampling']['resample_all_models_failed'])

    def test_missing_root_is_not_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            result=summarize(Path(tmp)/'absent')
            self.assertEqual(result['state'],'not_started')
            self.assertFalse((Path(tmp)/'absent').exists())

    def test_legacy_root_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp)/'results.json').write_text('[]')
            with self.assertRaises(RuntimeError): prepare_root(tmp)

    def test_manifest_can_resume_unchanged_but_not_different(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); expected=prepare_root(root)
            self.assertEqual(prepare_root(root),expected)
            self.assertEqual(verify_root(root),expected)
            expected['protocol_sha256']='old'
            (root/'RUN_PROTOCOL.json').write_text(json.dumps(expected))
            with self.assertRaises(RuntimeError): prepare_root(root)
            with self.assertRaises(RuntimeError): summarize(root)

    def test_completed_failure_is_not_retried_and_other_task_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); task={'route':1}
            (root/'result.json').write_text(json.dumps(self.row(task)))
            (root/'trajectory.npz').touch()
            self.assertFalse(completed_or_archive(root,'our_model',task)['success'])
            with self.assertRaises(RuntimeError): completed_or_archive(root,'our_model',{'route':2})

    def test_old_execution_result_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); row=self.row({}); row['execution_mode']='wall_clock_differential_drive_rgb_online_v1'
            (root/'result.json').write_text(json.dumps(row)); (root/'trajectory.npz').touch()
            with self.assertRaises(RuntimeError): completed_or_archive(root,'our_model',{})

    def test_all_failed_remain_in_paired_denominator(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); prepare_root(root)
            attempt=root/'groups/group_0000_hm3d/attempt_000'
            attempt.mkdir(parents=True)
            task=dict(**stamp(),source='hm3d',route_index=0)
            (attempt/'tasks.json').write_text(json.dumps([task]))
            for method in METHODS:
                out=attempt/'route_00'/method; out.mkdir(parents=True)
                row=self.row(task); row['method']=method; row['executor_policy']=policy_for(method)
                (out/'result.json').write_text(json.dumps(row)); (out/'trajectory.npz').touch()
            result=summarize(root)
            self.assertEqual(result['all_failed_routes'],1)
            self.assertEqual(result['fully_compared_routes'],1)
            self.assertEqual(result['any_success_routes'],0)
            self.assertEqual(result['paired_fully_compared_stats']['our_model']['success_rate'],0.)
            self.assertIsNone(result['conditional_any_success_stats']['our_model']['success_rate'])
            self.assertIsNone(result['model_stats']['our_model']['raw_length_ratio_own_successes_mean'])


if __name__ == '__main__': unittest.main()
