"""Fail closed: BIT* must receive numerical fused SHM, never Single aliases."""
import math
from shm_support import POLICY


class InvalidMetricExtent(RuntimeError):
    """A numerical model prediction rejection, not a fusion/driver failure."""
    def __init__(self, extent):
        self.extent = extent
        super().__init__(f'Invalid SHM metric extent: {extent!r}; expected [0.5, 30] m')


def planner_payload(response):
    if response.get('planner_bev_source')!='shm_latest_wins' or not response.get('fusion_contract',{}).get('enabled'):
        raise RuntimeError('SHM benchmark received Single/no-fusion response')
    if response['fusion_contract'].get('planner_support_policy') != POLICY:
        raise RuntimeError('SHM numeric FOV-boundary validity patch missing')
    if response['fusion_contract'].get('planner_support_inset_pixels') != 2:
        raise RuntimeError('SHM planner support inset differs from frozen protocol')
    count=response['history_frame_count']; seqs=response['history_frame_seqs']
    if not 1<=count<=10 or len(seqs)!=count or any(b<=a for a,b in zip(seqs,seqs[1:])):
        raise RuntimeError('Invalid SHM FIFO history')
    if seqs[-1]!=response['frame_seq']: raise RuntimeError('SHM map not anchored at latest frame')
    extent=float(response['shm_metric_extent_m'])
    if not math.isfinite(extent) or not .5<=extent<=30: raise InvalidMetricExtent(extent)
    return dict(model_single_semantic_png_base64=response['hard_merged_semantic_png_base64'],
        planner_occupancy_probability_u16_png_base64=response['shm_occupancy_probability_u16_png_base64'],
        planner_navigation_confidence_u16_png_base64=response['shm_navigation_confidence_u16_png_base64'],
        single_metric_extent_m=extent)
