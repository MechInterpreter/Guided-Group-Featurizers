"""Execution-only seed21 wrapper; preserve native fit identities and main receipts."""
import argparse,ast,copy,hashlib,json
from pathlib import Path
import all75_run as native

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def pilot_function():
    source=Path(native.__file__);tree=ast.parse(source.read_text(encoding='utf8'))
    function=copy.deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run'))
    counts=dict(seed_loop=0,expected_count=0,receipt=0,receipt_name=0)
    class ExecutionOnly(ast.NodeTransformer):
        def visit_For(self,node):
            if isinstance(node.target,ast.Name) and node.target.id=='seed' and isinstance(node.iter,ast.List) and [n.value for n in node.iter.elts]==[21,22,23]:
                node.iter=ast.List(elts=[ast.Constant(21)],ctx=ast.Load());counts['seed_loop']+=1
            return self.generic_visit(node)
        def visit_Assign(self,node):
            if any(isinstance(t,ast.Name) and t.id=='expected' for t in node.targets):
                node.value=ast.parse("2 if options['regime']=='full_category' else 5",mode='eval').body;counts['expected_count']+=1
            if any(isinstance(t,ast.Name) and t.id=='receipt' for t in node.targets):
                counts['receipt']+=1
                for kw in node.value.keywords:
                    if kw.arg=='reused_endpoints':kw.value=ast.parse("[r for r in reused if r['seed']==21]",mode='eval').body
                node.value.keywords.extend([
                    ast.keyword(arg='scope',value=ast.Constant('seed21_pilot')),
                    ast.keyword(arg='required_seeds',value=ast.List(elts=[ast.Constant(21)],ctx=ast.Load())),
                    ast.keyword(arg='full_three_seed_study_complete',value=ast.Constant(False)),
                    ast.keyword(arg='execution_revision_sha256',value=ast.Name(id='_PILOT_REVISION_SHA',ctx=ast.Load())),
                    ast.keyword(arg='pilot_amendment_sha256',value=ast.Name(id='_PILOT_AMENDMENT_SHA',ctx=ast.Load()))])
            return self.generic_visit(node)
        def visit_Constant(self,node):
            if node.value=='TRAINING_COMPLETE.json':counts['receipt_name']+=1;return ast.copy_location(ast.Constant('PILOT_SEED21_COMPLETE.json'),node)
            if node.value=='ENDPOINT_PROGRESS.json':return ast.copy_location(ast.Constant('PILOT_SEED21_PROGRESS.json'),node)
            if node.value=='TRAINING COMPLETE':return ast.copy_location(ast.Constant('SEED21 PILOT TRAINING COMPLETE'),node)
            return node
    function=ExecutionOnly().visit(function);module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
    if counts!=dict(seed_loop=1,expected_count=1,receipt=1,receipt_name=2):raise ValueError('Original orchestrator changed; review execution adapter instead of guessing: '+str(counts))
    namespace=dict(native.__dict__,_PILOT_REVISION_SHA=sha(__file__),_PILOT_AMENDMENT_SHA=sha(Path(__file__).with_name('PILOT_AMENDMENT.json')))
    exec(compile(module,str(source)+'::seed21-execution-only','exec'),namespace)
    return namespace['run'],counts

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);a=p.parse_args()
    options=json.loads(Path(a.config).read_text());pins=json.loads(Path(__file__).with_name('EVALUATION_PINS.json').read_text())
    if options['experiment_sha']!=pins['training_source_sha256']:raise ValueError('Preserve original model-store source identity for exact resume')
    print('SEED21-FIRST PILOT: same native fits/settings; seeds22-23 postponed, original main specification unchanged.',flush=True)
    function,_=pilot_function();function(options)
