"""CPU-only HTTP response boundary tests; no model imports or GPU execution."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch, Mock
from urllib.error import HTTPError
import test_shm_contract


def module_with_response(response):
    worker=types.ModuleType('procthor_benchmark_worker')
    worker.CAMERA_HEIGHT_M=.5
    worker.post_json=Mock(return_value=response)
    worker.decode_bev_response=Mock(side_effect=AssertionError('Invalid extent reached decoder'))
    spec=importlib.util.spec_from_file_location('_rejection_episode_test',Path(__file__).with_name('episode.py'))
    module=importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules,{'procthor_benchmark_worker':worker}):
        spec.loader.exec_module(module)
    return module,worker


class PredictionRejectionTests(unittest.TestCase):
    def response(self):
        row=test_shm_contract.ShmContractTests().response()
        row.update(shm_metric_extent_m=31.,shm_historical_owner_pixels=0,
                   shm_total_supported_pixels=100,inference_seconds=.25)
        return row

    def test_bad_scale_returns_no_path_and_preserves_evidence(self):
        m,w=module_with_response(self.response())
        with tempfile.TemporaryDirectory() as folder:
            inf=m.Inference('our_model','single','base',folder)
            result=inf.predict(['image'],[0,2],None)
            self.assertEqual(result['path_metric_m'],[])
            self.assertEqual(result['planning_failure']['code'],'invalid_predicted_metric_extent')
            self.assertEqual(result['metric_extent_m'],31.)
            self.assertEqual(result['planner_seconds'],0.)
            self.assertFalse(w.decode_bev_response.called)
            rows=(Path(folder)/'prediction_rejections.jsonl').read_text().splitlines()
            self.assertEqual(json.loads(rows[0])['shm_metric_extent_m'],31.)

    def test_fusion_contract_error_still_fatal(self):
        row=self.response();row['fusion_contract']['enabled']=False
        m,w=module_with_response(row)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaisesRegex(RuntimeError,'Single/no-fusion'):
                m.Inference('our_model','single','base',folder).predict(['image'],[0,2],None)
            self.assertFalse((Path(folder)/'prediction_rejections.jsonl').exists())

    def test_transport_or_cuda_failure_is_never_converted(self):
        for error in (RuntimeError('CUDA out of memory'),HTTPError('single',400,'bad',{},None)):
            m,w=module_with_response(self.response());w.post_json.side_effect=error
            with tempfile.TemporaryDirectory() as folder:
                with self.assertRaises(type(error)):
                    m.Inference('our_model','single','base',folder).predict(['image'],[0,2],None)


if __name__=='__main__':unittest.main()
