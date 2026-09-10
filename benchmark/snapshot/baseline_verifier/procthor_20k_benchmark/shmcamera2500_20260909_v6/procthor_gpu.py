"""Process-local Vulkan PCI selection; never change shared AI2-THOR mappings."""
import os
from pathlib import Path
import re
import subprocess


def pci_tag(bus):
    domain, slot, function = bus.strip().lower().split(':')
    return f'pci-{int(domain, 16):04x}_{slot}_{function.replace(".", "_")}'


def configure(runtime_root, gpu_index, expected_uuid):
    actual, bus = subprocess.check_output([
        'nvidia-smi', '-i', str(gpu_index), '--query-gpu=uuid,pci.bus_id',
        '--format=csv,noheader,nounits'], text=True, timeout=15).strip().split(',')
    if actual.strip().lower() != expected_uuid.lower():
        raise RuntimeError('GPU index/UUID changed before ProcTHOR startup')
    # Vendor/device IDs are identical across eight cards: select the PCI address.
    for key in ('DISPLAY', 'WAYLAND_DISPLAY', 'NODEVICE_SELECT', 'MESA_VK_DEVICE_SELECT'):
        os.environ.pop(key, None)
    os.environ['DRI_PRIME'] = pci_tag(bus)
    os.environ['MESA_VK_DEVICE_SELECT_FORCE_DEFAULT_DEVICE'] = '1'
    os.environ['CUDA_VISIBLE_DEVICES'] = actual.strip()
    os.environ['NVIDIA_VISIBLE_DEVICES'] = actual.strip()
    canonical_bus = f'{int(bus.strip().split(":")[0],16):04x}:'+':'.join(bus.strip().lower().split(':')[1:])
    info = Path('/proc/driver/nvidia/gpus')/canonical_bus/'information'
    minor = int(re.search(r'Device Minor:\s*(\d+)', info.read_text()).group(1))
    nodes = [f'/dev/nvidia{minor}']
    for name in ('nvidiactl','nvidia-uvm','nvidia-uvm-tools','nvidia-modeset'):
        if Path('/dev',name).exists(): nodes.append('/dev/'+name)
    for node in Path('/sys/class/drm').iterdir():
        if (node/'device').exists() and (node/'device').resolve().name.lower()==canonical_bus:
            candidate=Path('/dev/dri')/node.name
            if candidate.exists(): nodes.append(str(candidate))
    if not any('/renderD' in node for node in nodes):
        raise RuntimeError('No DRM render node matches selected PCI GPU')
    os.environ['REALTIME_GPU_DEVICE_NODES']=__import__('json').dumps(nodes)
    probe = Path(runtime_root)/'tools/vulkan-tools/root/usr/bin/vulkaninfo'
    output = subprocess.check_output([str(probe), '--summary'], text=True,
                                     stderr=subprocess.PIPE, timeout=20)
    uuids = re.findall(r'deviceUUID\s*=\s*([a-fA-F0-9-]+)', output)
    if [u.lower() for u in uuids] != [actual.strip()[4:].lower()]:
        raise RuntimeError(f'Vulkan is not exclusively bound to {actual}: {uuids}')
    return dict(gpu_index=gpu_index, uuid=actual.strip(), pci_bus=bus.strip(),
                dri_prime=os.environ['DRI_PRIME'], vulkan_visible_uuids=uuids,
                device_minor=minor, exposed_device_nodes=nodes,
                isolation='unprivileged bubblewrap private /dev for Unity only')


def assert_process_gpu_binding(pid, expected_uuid):
    """Wrapper PID may differ from Unity child; validate the full owned tree."""
    from dispatch import NvmlEvents, descendants
    events=NvmlEvents()
    try:
        rows=[r for r in events.processes() if r['pid'] in descendants(pid)]
        if not rows or any(r['uuid'].lower()!=expected_uuid.lower() for r in rows):
            raise RuntimeError(f'Unity process tree GPU binding mismatch: {rows}')
    finally:
        events.close()


def isolated_cloud_controller(runtime_root, *, physical_gpu_index, **kwargs):
    from ai2thor.controller import Controller
    from ai2thor.platform import CloudRendering

    class SingleVulkanController(Controller):
        @property
        def base_dir(self):
            return str(Path(runtime_root)/'runtime_home/.ai2thor')

        @property
        def log_dir(self):
            return str(Path(os.environ['REALTIME_UNITY_LOG_DIR']))

        def unity_command(self, width, height, headless):
            # UUID CUDA masks contain no integer indices; gpu_device=None avoids
            # the old node-wide CUDA-to-Vulkan-index cache. Visible Vulkan=one.
            command = super().unity_command(width, height, headless)
            import json
            wrapped=['bwrap','--unshare-user','--die-with-parent','--bind','/','/',
                     '--dev','/dev','--dir','/dev/dri','--bind','/dev/shm','/dev/shm']
            for node in json.loads(os.environ['REALTIME_GPU_DEVICE_NODES']):
                wrapped += ['--dev-bind',node,node]
            return wrapped + command + ['-force-device-index', '0', '-logFile',
                                        str(Path(self.log_dir)/'Player.log')]

    return SingleVulkanController(platform=CloudRendering, gpu_device=None, **kwargs)
