import bpy, time
TARGET=10;TOL=0.05
t=bpy.data.objects.get("TimerText")

class OT(bpy.types.Operator):
 bl_idname="view3d.tg";bl_label="Timer"
 run=False;start=0;timer=None
 def modal(self,c,e):
  if e.type=='SPACE'and e.value=='PRESS':
   if not self.run:
    self.start=time.perf_counter();self.run=True;t.data.body="..."
   else:
    self.run=False
    d=(time.perf_counter()-self.start)-TARGET
    t.data.body=("🎯"if abs(d)<=TOL else "❌")+f" Δ{d:+.3f}s"
  if e.type in{'ESC','RIGHTMOUSE'}:
   c.window_manager.event_timer_remove(self.timer)
   return{'CANCELLED'}
  return{'RUNNING_MODAL'}
 def invoke(self,c,e):
  if not t:
   self.report({'ERROR'},"Need TimerText");return{'CANCELLED'}
  self.timer=c.window_manager.event_timer_add(0.1,window=c.window)
  c.window_manager.modal_handler_add(self)
  t.data.body="Press SPACE"
  return{'RUNNING_MODAL'}

try:bpy.utils.unregister_class(OT)
except:pass
bpy.utils.register_class(OT)
bpy.ops.view3d.tg('INVOKE_DEFAULT')
