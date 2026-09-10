import unittest
from shm_contract import planner_payload, InvalidMetricExtent
from shm_support import POLICY, trusted_latest_mask
import numpy as np


class ShmContractTests(unittest.TestCase):
    def response(self):
        return dict(planner_bev_source='shm_latest_wins',fusion_contract=dict(enabled=True,planner_support_policy=POLICY,planner_support_inset_pixels=2),
            history_frame_count=2,history_frame_seqs=[1,2],frame_seq=2,shm_metric_extent_m=6.5,
            hard_merged_semantic_png_base64='FUSED_SEM',shm_occupancy_probability_u16_png_base64='FUSED_P',
            shm_navigation_confidence_u16_png_base64='FUSED_C',model_single_semantic_png_base64='WRONG_SINGLE',
            planner_occupancy_probability_u16_png_base64='WRONG_SINGLE_P')

    def test_only_fused_fields_enter_planner(self):
        out=planner_payload(self.response())
        self.assertEqual(out['model_single_semantic_png_base64'],'FUSED_SEM')
        self.assertEqual(out['planner_occupancy_probability_u16_png_base64'],'FUSED_P')
        self.assertEqual(out['planner_navigation_confidence_u16_png_base64'],'FUSED_C')

    def test_disabled_fusion_rejected(self):
        row=self.response();row['fusion_contract']['enabled']=False
        with self.assertRaises(RuntimeError):planner_payload(row)

    def test_bad_model_extent_has_typed_rejection_without_clamping(self):
        for extent in (.49, 30.01, 0., -1., float('inf'), float('nan')):
            row=self.response();row['shm_metric_extent_m']=extent
            with self.assertRaises(InvalidMetricExtent):planner_payload(row)
        for extent in (.5, 6.5, 30.):
            row=self.response();row['shm_metric_extent_m']=extent
            self.assertEqual(planner_payload(row)['single_metric_extent_m'],extent)

    def test_contract_error_is_not_numerical_rejection(self):
        row=self.response();row['fusion_contract']['enabled']=False
        row['shm_metric_extent_m']=float('nan')
        with self.assertRaises(RuntimeError) as caught:planner_payload(row)
        self.assertNotIsInstance(caught.exception,InvalidMetricExtent)

    def test_missing_fused_fields_does_not_fallback(self):
        row=self.response();del row['shm_occupancy_probability_u16_png_base64']
        with self.assertRaises(KeyError):planner_payload(row)

    def test_old_anchor_rejected(self):
        row=self.response();row['frame_seq']=3
        with self.assertRaises(RuntimeError):planner_payload(row)

    def test_old_unpatched_support_rejected(self):
        row=self.response();del row['fusion_contract']['planner_support_policy']
        with self.assertRaises(RuntimeError):planner_payload(row)

    def test_untrusted_border_does_not_become_free_or_overwrite_old(self):
        valid=np.ones((3,3));interior=np.zeros((3,3));interior[1,1]=1
        write=trusted_latest_mask(valid,interior)
        occupancy=np.full((3,3),.5);support=np.zeros((3,3),bool)
        incoming=np.ones((3,3))
        occupancy[write]=incoming[write];support[write]=True
        self.assertEqual(np.sum(support),1)
        self.assertEqual(occupancy[0,0],.5);self.assertFalse(support[0,0])
        occupancy[0,0]=1.;support[0,0]=True
        occupancy[write]=0.
        self.assertEqual(occupancy[0,0],1.)
        self.assertEqual(occupancy[1,1],0.)

    def test_interior_never_extends_predicted_support(self):
        valid=np.zeros((3,3));interior=np.ones((3,3))
        self.assertFalse(trusted_latest_mask(valid,interior).any())
        with self.assertRaises(ValueError):trusted_latest_mask(valid,np.ones((2,2)))
