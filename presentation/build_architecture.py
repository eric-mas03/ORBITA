"""Rebuild the editable diagrams.net companion to architecture.svg (standard library only)."""
from pathlib import Path
import xml.etree.ElementTree as ET

HERE=Path(__file__).resolve().parent
root=ET.Element('mxfile',{'host':'app.diagrams.net','version':'26.0.0','type':'device','compressed':'false'})
diagram=ET.SubElement(root,'diagram',{'id':'orbita-shared-memory','name':'ORBITA research architecture'})
model=ET.SubElement(diagram,'mxGraphModel',{'dx':'1600','dy':'900','grid':'1','gridSize':'10','guides':'1','tooltips':'1','connect':'1','arrows':'1','fold':'1','page':'1','pageScale':'1','pageWidth':'1600','pageHeight':'900','math':'0','shadow':'0','background':'#f5f8f5'})
graph=ET.SubElement(model,'root');ET.SubElement(graph,'mxCell',{'id':'0'});ET.SubElement(graph,'mxCell',{'id':'1','parent':'0'})

def vertex(identifier,label,x,y,width,height,fill='#ffffff',stroke='#d8e4dc',font=20,style=''):
    cell=ET.SubElement(graph,'mxCell',{'id':identifier,'value':label,'vertex':'1','parent':'1','style':f'rounded=1;whiteSpace=wrap;html=1;arcSize=12;fillColor={fill};strokeColor={stroke};fontColor=#203b36;fontFamily=Segoe UI;fontSize={font};align=center;verticalAlign=middle;spacing=12;'+style})
    ET.SubElement(cell,'mxGeometry',{'x':str(x),'y':str(y),'width':str(width),'height':str(height),'as':'geometry'})
    return identifier

def text(identifier,label,x,y,width,height,font=20,colour='#203b36',bold=False):
    return vertex(identifier,label,x,y,width,height,'none','none',font,f'align=left;spacing=0;fontColor={colour};fontStyle={1 if bold else 0};')

def edge(identifier,source,target,colour='#9db8ad',label=''):
    cell=ET.SubElement(graph,'mxCell',{'id':identifier,'value':label,'edge':'1','parent':'1','source':source,'target':target,'style':f'edgeStyle=orthogonalEdgeStyle;rounded=1;orthogonalLoop=1;jettySize=auto;html=1;endArrow=block;endFill=1;strokeColor={colour};strokeWidth=2;dashed=1;dashPattern=5 5;fontFamily=Segoe UI;fontSize=12;fontColor=#627970;'})
    ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})

text('brand','ORBITA · RESEARCH ARCHITECTURE',64,45,1100,30,13,'#69877b',True)
text('title','From observations to an informed forecast.',64,94,1450,56,46,'#203b36',True)
text('subtitle','A shared memory of change, with a separate forecast for each source.',64,151,1450,34,21,'#637c74')
for identifier,label,x in [('observe','01 · OBSERVE',64),('encode','02 · FIND PATTERNS',370),('memory-stage','03 · SHARED MEMORY',651),('query-stage','04 · LOOK AHEAD',1006),('output-stage','05 · SHOW THE RANGE',1250)]:text(identifier,label,x,207,290,32,12,'#69877b',True)

sources=[('heart','<b>Heart measurements</b><br><font size="3">NASA ground analogs<br>Real visits · original cadence</font>',248,'#70ab91','#e9f2ec'),('radiation','<b>Station radiation</b><br><font size="3">NASA RadLab · ISS<br>Instrument environment</font>',390,'#8aafd1','#e9eff7'),('wearable','<b>Glucose and pulse</b><br><font size="3">Earth wearable study<br>Known history + meal records</font>',532,'#cf9aa1','#f7ebed')]
for name,label,y,colour,fill in sources:
    vertex('source-'+name,label,64,y,240,114,stroke=colour,font=20)
    encoder={'heart':'Heart encoder','radiation':'Radiation encoder','wearable':'Wearable encoders'}[name]
    vertex('encoder-'+name,f'<b>{encoder}</b><br><font size="2">Values · presence · age</font>',370,y+20,184,74,fill,colour,19)
    edge('signal-'+name,'source-'+name,'encoder-'+name,colour)

vertex('shared-memory','',651,248,292,398,'#e3f1e9','#bfd5c8')
text('memory-title','Shared temporal<br>representation',681,268,236,64,23,'#203b36',True)
text('latent-subtitle','A space for patterns',688,330,230,27,17,'#627970')
point_coordinates=[(718,399,'#70ab91'),(779,367,'#8aafd1'),(864,406,'#cf9aa1'),(841,483,'#70ab91'),(765,517,'#8aafd1'),(704,470,'#cf9aa1'),(792,442,'#4e8873')]
for index,(cx,cy,colour) in enumerate(point_coordinates):
    vertex('latent-'+str(index),'',cx-10,cy-10,20,20,colour,colour,12,'ellipse;aspect=fixed;')
for index,(a,b) in enumerate([(0,1),(1,2),(2,3),(3,4),(4,5),(5,0),(0,6),(1,6),(2,6),(3,6),(4,6),(5,6)]):
    cell=ET.SubElement(graph,'mxCell',{'id':'latent-link-'+str(index),'edge':'1','parent':'1','source':'latent-'+str(a),'target':'latent-'+str(b),'style':'endArrow=none;strokeColor=#9cb9ac;strokeWidth=1;opacity=55;'})
    ET.SubElement(cell,'mxGeometry',{'relative':'1','as':'geometry'})
text('memory-explainer','Learn changes over time',685,553,240,28,17,'#627970')
text('memory-tech','Shared causal Transformer<br>Source identity remains explicit',684,590,242,43,13,'#6b817a')
for name,_,_,colour,_ in sources:edge('to-memory-'+name,'encoder-'+name,'shared-memory',colour)

vertex('horizon-queries','<b>Choose a horizon</b><br><br>1 h · 4 h · 8 h · 24 h<br><font size="2">Continuous sources</font><br><br>Or the next visit<br><font size="2">For sparse echo measurements</font>',1006,309,187,276,font=20)
edge('memory-to-query','shared-memory','horizon-queries','#4e8873')
vertex('forecast-heads','<b>A forecast per source</b><br><font size="3">With visible uncertainty</font>',1250,248,286,398,font=22,style='verticalAlign=top;spacingTop=28;')
text('quantile-values','Lower estimate — central estimate — upper estimate',1275,380,235,100,22,'#43856e',True)
text('head-tech','Source-specific residual heads<br>Blend selected on validation data',1278,588,232,47,13,'#6b817a')
edge('query-to-head','horizon-queries','forecast-heads','#4e8873')
text('integrity','Different studies are never treated as one astronaut. Missing measurements stay masked.',64,678,1472,37,17,'#627970')
vertex('next-step','<font size="2">THE NEXT STEP</font><br><b>More people. More frequent measurements. Better evidence for prediction quality.</b>',64,743,1472,103,'#e7eee8','#e7eee8',27,style='align=left;spacingLeft=26;')

ET.indent(root,space='  ')
ET.ElementTree(root).write(HERE/'architecture.drawio',encoding='utf-8',xml_declaration=True)
print('Wrote architecture.drawio')
