"""Tables, scientific figures and a Russian article addendum from frozen outcomes."""
import csv,hashlib
from collections import defaultdict
from utils import ROOT,read_json,read_rows,write_json

def main():
    assert (ROOT/'results/final/access_log.json').exists()
    f=read_json(ROOT/'freeze.json');p=read_json(ROOT/'protocol.json');rows=read_json(ROOT/'results/final/comparisons.json');selected=rows[0]
    paper=ROOT/'paper';paper.mkdir(exist_ok=True)
    attempts=read_json(ROOT/'results/attempts.json')
    with (paper/'validation_attempts.csv').open('w',newline='') as fh:
        fields=['candidate','family','validation_correct','wrong_to_right','right_to_wrong','changed_scalars','delta_l2']
        w=csv.DictWriter(fh,fieldnames=fields);w.writeheader()
        for r in attempts:
            spec=read_json(ROOT/'results'/r['id']/'edit_spec.json')
            w.writerow({'candidate':r['id'],'family':spec['family'],**{k:r[k] for k in fields[2:]}})
    tex=[r'\begin{tabular}{lrrrrrr}',r'\hline',r'Condition & $N$ & Before (\%) & After (\%) & $\Delta$ pp & W$\to$R & R$\to$W \\',r'\hline']
    table=['| Условие | N | До, % | После, % | Δ, п.п. | Исправлено | Испорчено | 95% ДИ Δ |','|---|---:|---:|---:|---:|---:|---:|---|']
    for r in rows:
        lo,hi=r['paired_stratified_95ci_pp']
        tex.append(f"{r['model'].replace('_',chr(92)+'_')} & {r['n']} & {100*r['before_accuracy']:.3f} & {100*r['after_accuracy']:.3f} & {r['delta_pp']:+.3f} & {r['wrong_to_right']} & {r['right_to_wrong']} "+r'\\')
        table.append(f"| {r['model']} | {r['n']} | {100*r['before_accuracy']:.3f} | {100*r['after_accuracy']:.3f} | {r['delta_pp']:+.3f} | {r['wrong_to_right']} | {r['right_to_wrong']} | [{lo:+.3f}; {hi:+.3f}] |")
    tex.extend([r'\hline',r'\end{tabular}']);(paper/'results_table.tex').write_text('\n'.join(tex)+'\n')
    same=selected.get('same_token_length');ci=selected['paired_stratified_95ci_pp']
    positive=ci[0]>0 and bool(same and same['paired_stratified_95ci_pp'][0]>0)
    conclusion='Заранее заданный совместный критерий положительного результата выполнен на этих тестах.' if positive else 'Заранее заданный совместный критерий положительного результата не выполнен.'
    equal_text='Подмножество одинаковой токенной длины отсутствует.'
    if same:
        lo,hi=same['paired_stratified_95ci_pp']
        equal_text=f"На заранее объявленном подмножестве одинаковой токенной длины N={same['n']}: {same['before_correct']} → {same['after_correct']}, Δ={same['delta_pp']:+.3f} п.п., 95% ДИ [{lo:+.3f}; {hi:+.3f}]. Исправлено {same['wrong_to_right']}, испорчено {same['right_to_wrong']}."
    selected_spec=read_json(ROOT/'edit_spec.json')
    technical=ROOT/'results/technical_failures/alias_check_before_validation/failure.json'
    technical_note=''
    if technical.exists():technical_note='До первой полной оценки кандидата исправлена ошибочная ссылка на имя связанного embedding в проверочном коде. Сбой произошёл до validation; сохранены traceback/исходный оценщик, спецификация и протокол до исправления. Baseline повторён: открытые прогнозы точно совпали, совпали и хеши файлов закрытых прогнозов без просмотра их исходов. Это техническая поправка проверки, а не улучшение весов; завершено ровно 12 полных оценок validation.'
    draft=f'''# Дополнительный эксперимент: увеличение охвата прямой правки весов TinyStories

Этот раздел дополняет опыты на сумматоре и Glimmer. Задачи и поднаборы различаются, поэтому абсолютные проценты не являются рейтингом моделей.

## Модель и постановка

Использован готовый checkpoint roneneldan/TinyStories-1M, revision 77f1b168e219585646439073245fe87e56b3023e. Фактическое общее число независимых параметров — 3 745 984, а не 1 миллион; из них 398 336 в восьми трансформерных блоках. Архитектура GPT-Neo: hidden 64, 16 голов, FFN 256, GELU, обучаемые позиции, чередование глобального и локального внимания, связанный readout/embedding. Источник был обучен авторами на коротких историях; наш этап не запускает обучение, градиенты, оптимизатор, SFT, LoRA, регрессию, дистилляцию или случайный поиск содержательных кандидатов.

В отличие от прежних правок 16 и 52 скаляров, здесь проверяются целые проекции всех восьми блоков: 32 768–197 632 изменённых значений. Это увеличение охвата, но формулы всё ещё имеют несколько групповых коэффициентов. Большое число изменённых скаляров не доказывает программирование стольких независимых вычислений или появление нового алгоритма.

## Бенчмарк и контроль

До запуска модели выбраны восемь других разделов BLiMP: anaphor_number_agreement, determiner_noun_agreement_2, regular_plural_subject_verb_agreement_2, irregular_plural_subject_verb_agreement_2, irregular_past_participle_verbs, distractor_agreement_relational_noun, passive_1 и intransitive. Они не совпадают с поднабором Glimmer. После удаления точных повторов предложений сформированы diagnostic {p['datasets']['diagnostic']['n']}, validation {p['datasets']['validation']['n']} и test {p['datasets']['test']['n']}. Seeds и SHA-256 зафиксированы до правок.

Основная оценка — предпочтение грамматически правильного предложения по суммарной причинной лог-вероятности с BOS, без EOS и без нормализации длины; ничья считается ошибкой. Это стандартная teacher-forced likelihood-оценка минимальных пар, а не генерация ответа. Чтобы ограничить интерпретацию через длину строки, положительный результат заранее требует нижней границы 95% парного стратифицированного bootstrap-интервала выше нуля и на полном test, и на его подмножестве равной токенной длины. Во всех случаях сообщаются регрессии и разбивка по разделам.

Веса и архитектура загружаются строго. Старые фиксированные mask-buffers проверены на совпадение с официальным конструктором; связанная отсутствующая голова восстанавливается из исходного embedding. Эти операции совместимости предшествуют baseline и не считаются правкой весов. Сохранение/загрузка без изменения, откат, ties, buffers, dtype, shapes, конечность и сохранность вне маски проверяются; финальный инференс — новый процесс без диагностических вмешательств. Оригинальный checkpoint сохранён неизменным.

Исходный test оценивается отдельным процессом и не просматривается при выборе. До открытия фиксируются checkpoint, edit_spec, код и протокол. Бюджет — 12 полных оценок validation; выбирается максимум правильных пар, затем минимум регрессий, изменённых значений и нормы diff. Три случайных контроля имеют ту же фактическую маску и L2-норму с погрешностью float32 и заранее фиксированные seeds; лучший seed не отбирается.

{technical_note}

## Гипотезы и выбранная правка

Baseline: 672/1024 (65,625%) diagnostic и 1334/2048 (65,137%) validation. Проверены три содержательных семейства: gain attention-output всех блоков (0,9; 1,1; 1,2), gain FFN-output всех блоков (0,8; 0,9; 1,1), gain query-проекций (0,8; 1,2; 1,5). Первое меняет вклад контекста относительно residual, второе — вклад FFN, третье — концентрацию существующего внимания. Нормы активаций и диагностические ошибки мотивируют, но не доказывают причинную гипотезу. Три комбинации используют лучшие одиночные коэффициенты по заранее заданному правилу, без новых величин.

По validation выбран {f['selected']['id']}, семейство {selected_spec['family']}. Изменено {f['selected']['changed_scalars']} / 3 745 984 значений ({100*f['selected']['fraction']:.3f}% всех параметров; {100*f['selected']['changed_scalars']/398336:.3f}% параметров блоков). L2-норма изменения {f['selected']['delta_l2']:.8f}, относительная {f['selected']['relative_l2']:.8f}. Точные коэффициенты и маска — в edit_spec.json; все кандидаты и их регрессии сохранены. На validation выбранная правка дала {f['selected']['validation_correct']}/2048, исправив {f['selected']['wrong_to_right']} и испортив {f['selected']['right_to_wrong']} исходных ответов.

## Финальные результаты

{chr(10).join(table)}

{equal_text}

{conclusion}

Число дискордантных пар и точные условные двусторонние p сохранены как дополнительные проверки чувствительности; они не заменяют зарегистрированный критерий и не используются для нового выбора весов. Bootstrap — 20 000 реплик с фиксированным seed, внутри грамматических разделов с исходными весами страт. Отдельные разделы и случайные контроли описательные, без выбора лучшего по test. Пять коротких жадных продолжений — иллюстрации, не подтверждающий тест связности.

## Ограничения

Один checkpoint и один редактор при фиксированном бюджете не определяют эффект на другие модели и обучения. Поток обучения TinyStories не аудирован на точные пересечения с BLiMP; редкая лексика и конструкции BLiMP отличаются от домена простых историй. Шаблоны и словарь создают зависимости между парами. Bootstrap условен на выбранной правке и примерах, а не независимых предобучениях и адаптивном выборе моделей. Даже выполнение критерия на равной токенной длине не доказывает улучшения общей языковой способности. Нельзя выбирать новую правку по уже открытым финальным ошибкам и называть её независимым подтверждением.

Инференс: CPU, float32, один поток, официальный eager GPT-Neo, batch 16. Время измеряется после одинакового прогрева отдельно от загрузки и формирования отчёта. Единичные времена не составляют полноценный скоростной бенчмарк. Время применения правки не включает время рассуждений агента. Модель и код источника публично доступны, но явная лицензия весов в зафиксированной карточке/дереве не найдена; право их публичного распространения не утверждается. Предоставлены загрузчик исходника с хешами и декларативная спецификация изменения.

## Источники

1. Eldan R., Li Y. TinyStories: How Small Can Language Models Be and Still Speak Coherent English? 2023. https://arxiv.org/abs/2305.07759 . Модель https://huggingface.co/roneneldan/TinyStories-1M .
2. Warstadt A. et al. BLiMP: The Benchmark of Linguistic Minimal Pairs for English. TACL, 2020, 8:377–392. https://doi.org/10.1162/tacl_a_00321 . Данные CC-BY 4.0: https://github.com/alexwarstadt/blimp . Наш поднабор и разбиения изменяют схему оценки; одобрение авторов не предполагается.
'''
    (paper/'ARTICLE_ADDENDUM_RU.md').write_text(draft)
    before=read_rows(ROOT/'results/final/baseline/test.jsonl');after=read_rows(ROOT/'results/final/selected/test.jsonl');groups=defaultdict(list)
    for b,a in zip(before,after):
        d=len(b['good_tokens'])-len(b['bad_tokens']);key='same' if not d else 'good_shorter' if d<0 else 'good_longer'
        groups[key].append((b,a))
    write_json(paper/'length_analysis.json',{k:{'n':len(v),'before_correct':sum(b['correct'] for b,a in v),'after_correct':sum(a['correct'] for b,a in v)} for k,v in groups.items()})
    try:plots(paper,rows)
    except ImportError:print('Tables/addendum saved; install matplotlib for figures.')
    print('Paper artifacts:',paper)

def plots(paper,rows):
    import os
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(paper/f'{name}.{ext}',bbox_inches='tight',dpi=160)
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,3.7))
    for i,r in enumerate(rows):
        lo,hi=r['paired_stratified_95ci_pp'];d=r['delta_pp']
        ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=4,color='#176b87' if i==0 else '#777777')
    ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks(range(len(rows)),[r['model'] for r in rows]);ax.invert_yaxis()
    ax.set_xlabel('After − before pair accuracy (percentage points)');ax.set_title('TinyStories broad edits: paired 95% bootstrap intervals');save(fig,'paired_changes')
    r=rows[0];same=r.get('same_token_length')
    if same:
        fig,ax=plt.subplots(figsize=(7,2.8))
        for i,(name,v) in enumerate([('Full test',r),('Same token length',same)]):
            lo,hi=v['paired_stratified_95ci_pp'];d=v['delta_pp'];ax.errorbar(d,i,xerr=[[d-lo],[hi-d]],fmt='o',capsize=4,color='#176b87')
        ax.axvline(0,color='black',linestyle='--',linewidth=.8);ax.set_yticks([0,1],['Full test','Same token length']);ax.invert_yaxis()
        ax.set_xlabel('After − before pair accuracy (percentage points)');ax.set_title('Both tests are required by the preregistered success rule');save(fig,'joint_criterion')

if __name__=='__main__':main()
