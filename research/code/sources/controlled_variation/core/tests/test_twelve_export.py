import json,zipfile
from coco_adaptation import inputs
from coco_ggf import twelve_capacity as exp


def test_compact_export_keeps_results_and_excludes_rank_accumulators(tmp_path):
    s=inputs.JournalStore(tmp_path/'output',dict(test='export'),tmp_path/'local',False)
    ctx=dict(store=s,sources={})
    s.save('capacity_result',dict(complete=True,models=[]))
    s.write('test_means.csv',{'sample':'retained'})
    s.write('effective_rank/summary.json',dict(complete=True))
    s.write('effective_rank/accumulators/model/slot0.json',dict(private_resume_state=True))
    source=tmp_path/'source.zip'
    with zipfile.ZipFile(source,'w') as z:z.writestr('source.py','print(1)')
    path=exp.export(ctx,source)
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        assert 'results/test_means.csv' in z.namelist()
        assert 'results/effective_rank/summary.json' in z.namelist()
        assert not any('accumulators/' in n for n in z.namelist())
        assert json.loads(z.read('results/summary.json'))['rank_complete']
        hashes=json.loads(z.read('artifact_hashes.json'))
        assert set(hashes)==set(z.namelist())-{'artifact_hashes.json'}
        assert z.read('source.zip')==source.read_bytes()
