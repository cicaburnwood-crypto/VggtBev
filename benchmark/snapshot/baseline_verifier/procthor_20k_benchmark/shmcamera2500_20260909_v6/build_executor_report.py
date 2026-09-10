"""Render actual acceptance evidence, never synthetic model benchmark results."""
import argparse
import hashlib
import json
from pathlib import Path
import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from robot_contract import SPEC


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--trials',type=Path,nargs='+',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--write-acceptance',action='store_true',
                   help='Write a source-bound gate only after both live backend reports pass')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    suite=unittest.defaultTestLoader.discover(str(Path(__file__).parent), pattern='test_*.py')
    test_result=unittest.TextTestRunner(verbosity=1).run(suite)
    if not test_result.wasSuccessful():raise SystemExit('CPU regression failed')
    reports=[json.loads((root/'PASS.json').read_text()) for root in args.trials]
    for report in reports:
        if report.get('robot') != SPEC.metadata():
            raise SystemExit('Acceptance evidence uses an obsolete robot/executor contract')
        if len(report.get('routes', [])) != 5 or any(
                row['status'] != 'complete' for row in report['routes']):
            raise SystemExit('Live executor acceptance is incomplete')
    metrics=[row for report in reports for row in report['routes']]
    summary=dict(cpu_tests=test_result.testsRun,cpu_pass=True,live_routes=len(metrics),
        live_completed=sum(row['status']=='complete' for row in metrics),
        endpoint_error_max_m=max(row['endpoint_error_m'] for row in metrics),
        cross_track_error_max_m=max(row['cross_track_max_m'] for row in metrics),
        max_speed_m_s=max(row['max_speed_m_s'] for row in metrics),
        max_yaw_deg_s=max(row['max_yaw_deg_s'] for row in metrics),
        rgb_frames=sum(row['rgb_frames'] for row in metrics),
        min_camera_height_m=min(row['camera_height_min_m'] for row in metrics),
        max_camera_height_m=max(row['camera_height_max_m'] for row in metrics),
        measured_wall_seconds=sum(row['wall_seconds'] for row in metrics),
        nominal_motion_seconds=sum(row['motion_seconds'] for row in metrics),
        sources=[r.get('source','procthor') for r in reports],
        evidence=[str(root.resolve()) for root in args.trials],
        scope='executor-only GT command routes, NOT model performance',
        code_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('exact_executor.py','body_collision.py','robot_contract.py','virtual_episode.py')})
    (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(2,max(2,len(reports)),figsize=(14,9),squeeze=False)
    colors=['#1875D1','#C72C6B','#13805E','#A66C00','#7446AB']
    for column,(root,report) in enumerate(zip(args.trials,reports)):
        geometry=np.load(root/'geometry.npz')
        truth,lower,v=geometry['truth'],geometry['lower_bound'],float(geometry['voxel_size_m'])
        xmin,zmin=lower[[0,2]]-v/2
        extent=[xmin,xmin+truth.shape[1]*v,zmin,zmin+truth.shape[0]*v]
        ax=axes[0,column]
        ax.imshow(truth,cmap='gray',vmin=0,vmax=255,extent=extent,origin='upper',interpolation='nearest')
        tasks=json.loads((root/'tasks.json').read_text())
        for i,task in enumerate(tasks):
            path=np.asarray(task['world'])
            trace=np.load(root/f'route_{i:02d}'/'trajectory.npz')
            ax.plot(path[:,0],path[:,2],color=colors[i],lw=3,alpha=.4)
            ax.plot(trace['x'],trace['z'],color=colors[i],lw=1,ls='--',label=f'Route {i+1}')
            ax.scatter(path[0,0],path[0,2],marker='^',s=28,color=colors[i])
        ax.set_title(f"{report.get('source','ProcTHOR').upper()} | 5/5 GT routes completed",fontsize=12)
        ax.set_xlabel('World X (m)');ax.set_ylabel('World Z (m)');ax.legend(fontsize=8,loc='best')
        ax=axes[1,column]
        i=max(range(5),key=lambda i:report['routes'][i]['motion_seconds'])
        t=np.load(root/f'route_{i:02d}'/'trajectory.npz')
        mask=t['dt']>0
        velocity=np.divide(t['ds'],t['dt'],out=np.zeros_like(t['ds']),where=mask)
        angular=np.divide(np.abs(t['dyaw']),t['dt'],out=np.zeros_like(t['dyaw']),where=mask)/SPEC.yaw_rate_rad_s
        ax.plot(t['motion_t'],velocity,color='#1875D1',lw=1.2,label='Translation / 1 m/s')
        ax.plot(t['motion_t'],angular,color='#A66C00',lw=1.2,label=f'Angular speed / {SPEC.yaw_rate_deg_s:g} deg/s')
        ax.set_title(f'Route {i+1}: separate translation and in-place rotation',fontsize=11)
        ax.axhline(1,color='#B31B1B',ls='--',lw=1,label='Requested rate limit')
        ax.set_ylim(-.05,1.15);ax.set_xlabel('Motion time, excluding render pauses (s)')
        ax.set_ylabel('Fraction of translation / yaw rate limit');ax.legend(fontsize=8)
    if len(reports)==1:
        for ax in axes[:,1]:ax.axis('off')
        from PIL import Image
        axes[0,1].imshow(Image.open(args.trials[0]/'route_03'/'rgb_0000.jpg'))
        axes[0,1].set_title('Actual initial RGB | camera height 0.50 m',fontsize=12)
        axes[1,1].imshow(Image.open(args.trials[0]/'route_03'/'rgb_terminal.jpg'))
        axes[1,1].set_title('Actual terminal RGB | route 4',fontsize=12)
    fig.suptitle('Bounded executor acceptance — not a model benchmark',fontsize=15)
    fig.text(.5,.015,f"{len(metrics)} live GT routes | max endpoint error {summary['endpoint_error_max_m']:.1e} m | "
             f'1 m/s | {SPEC.yaw_rate_deg_s:g} deg/s | body {SPEC.width_m:.2f} x {SPEC.length_m:.2f} x {SPEC.height_m:.2f} m',ha='center',fontsize=10)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(args.output/'executor_acceptance.png',dpi=160)
    plt.close(fig)
    lines=['# 执行器验收结果','',
           '本报告是 GT 指令路线的执行器测试，不是模型成功率。','',
           f"CPU 测试：{summary['cpu_tests']}/{summary['cpu_tests']} 通过。",
           f"模拟器路线：{summary['live_completed']}/{summary['live_routes']} 完整执行。",'',
           '| 数据源 | 路线 | 长度(m) | 移动＋转向(s) | 实际耗时(s) | 终点误差(m) |',
           '|---|---:|---:|---:|---:|---:|']
    for report in reports:
        for i,row in enumerate(report['routes']):
            lines.append(f"| {report.get('source','ProcTHOR')} | {i+1} | {row['travelled_m']:.3f} | "
                f"{row['motion_seconds']:.3f} | {row['wall_seconds']:.3f} | {row['endpoint_error_m']:.2e} |")
    lines+=['',f'最大平移速度：1 m/s；最大角速度：{SPEC.yaw_rate_deg_s:g}°/s；相机高度：0.5 m。',
            '直角折点停下转向，严格保留原生折线；没有加速度约束，也不宣称真实动力学。',
            '渲染器实际相机位置、朝向每帧核对，误差阈值分别为 1e-5 m 和 1e-5 rad。',
            '表内的更小误差是执行器双精度轨迹的数值误差，并非仿真器资产的几何精度。',
            '无碰撞结论仅限这些 GT 测试路线；模型错误、网格资产误差不因此被排除。','',
            '![Acceptance](executor_acceptance.png)','']
    (args.output/'REPORT.md').write_text('\n'.join(lines))
    if args.write_acceptance:
        if not ({'procthor','hm3d'} <= set(summary['sources']) and
                summary['live_completed'] >= 10 and
                summary['endpoint_error_max_m'] < 1e-7 and
                summary['cross_track_error_max_m'] < 1e-7 and
                summary['max_speed_m_s'] <= SPEC.speed_m_s+1e-9 and
                summary['max_yaw_deg_s'] <= SPEC.yaw_rate_deg_s+1e-9):
            raise SystemExit('Insufficient live evidence for executor acceptance')
        Path(__file__).with_name('EXECUTOR_SIMULATOR_ACCEPTED.json').write_text(
            json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
