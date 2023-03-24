#!/usr/bin/env python3

import argparse
try:
	import Tkinter
except ImportError:
	import tkinter as Tkinter
from ruamel.yaml import YAML
import pathlib
import os
import subprocess
import re
import time
import threading

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--configpath",
		type=str,
		default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../config/crazyflies.yaml"),
		help="Path to the configuration .yaml file")
	parser.add_argument(
		"--stm32Fw",
		type=str,
		default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../crazyflie-firmware/cf2.bin"),
		help="Path to cf2.bin")
	parser.add_argument(
		"--nrf51Fw",
		type=str,
		default=os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../../../crazyflie2-nrf-firmware/cf2_nrf.bin"),
		help="Path to cf2_nrf.bin")
	args = parser.parse_args()

	if not os.path.exists(args.configpath):
		print("ERROR: Could not find yaml configuration file in configpath ({}).".format(args.configpath))
		exit()

	if not os.path.exists(args.stm32Fw):
		print("WARNING: Could not find STM32 firmware ({}).".format(args.stm32Fw))

	if not os.path.exists(args.nrf51Fw):
		print("WARNING: Could not find NRF51 firmware ({}).".format(args.nrf51Fw))

	def selected_cfs():
		nodes = {name: node for name, node in cfg["robots"].items() if widgets[name].checked.get()}
		return nodes

	def save():
		for name, node in cfg["robots"].items():
			if widgets[name].checked.get():
				node["enabled"] = True
			else:
				node["enabled"] = False
		with open(args.configpath, 'w') as outfile:
			yaml.dump(cfg, outfile)

	yaml = YAML()
	cfg = yaml.load(pathlib.Path(args.configpath))
	cfTypes = cfg["robot_types"]
	enabled = [name for name in cfg["robots"].keys() if cfg["robots"][name]["enabled"] == True]


	# compute absolute pixel coordinates from the initial positions
	positions = [node["initial_position"] for node in cfg["robots"].values()]
	DOWN_DIR = [-1, 0]
	RIGHT_DIR = [0, -1]
	def dot(a, b):
		return a[0] * b[0] + a[1] * b[1]
	pixel_x = [120 * dot(pos, RIGHT_DIR) for pos in positions]
	pixel_y = [120 * dot(pos, DOWN_DIR) for pos in positions]
	xmin, ymin = min(pixel_x), min(pixel_y)
	xmax, ymax = max(pixel_x), max(pixel_y)

	# construct the main window
	top = Tkinter.Tk()
	top.title('Crazyflie Chooser')

	# construct the frame containing the absolute-positioned checkboxes
	width = xmax - xmin + 50 # account for checkbox + text width
	height = ymax - ymin + 50 # account for checkbox + text height
	frame = Tkinter.Frame(top, width=width, height=height)

	class CFWidget(Tkinter.Frame):
		def __init__(self, parent, name):
			Tkinter.Frame.__init__(self, parent)
			self.checked = Tkinter.BooleanVar()
			checkbox = Tkinter.Checkbutton(self, variable=self.checked, command=save,
				padx=0, pady=0)
			checkbox.grid(row=0, column=0, sticky='E')
			nameLabel = Tkinter.Label(self, text=name, padx=0, pady=0)
			nameLabel.grid(row=0, column=1, sticky='W')
			self.batteryLabel = Tkinter.Label(self, text="", fg="#999999", padx=0, pady=0)
			self.batteryLabel.grid(row=1, column=0, columnspan=2, sticky='E')
			self.versionLabel = Tkinter.Label(self, text="", fg="#999999", padx=0, pady=0)
			self.versionLabel.grid(row=2, column=0, columnspan=2, sticky='E')

	# construct all the checkboxes
	widgets = {}
	for (id, node), x, y in zip(cfg["robots"].items(), pixel_x, pixel_y):
		w = CFWidget(frame, str(id))
		w.place(x = x - xmin, y = y - ymin)
		w.checked.set(id in enabled)
		widgets[id] = w

	# dragging functionality - TODO alt-drag to deselect
	drag_start = None
	drag_startstate = None

	def minmax(a, b):
		return min(a, b), max(a, b)

	def mouseDown(event):
		global drag_start, drag_startstate
		drag_start = (event.x_root, event.y_root)
		drag_startstate = [cf.checked.get() for cf in widgets.values()]

	def mouseUp(event):
		save()

	def drag(event, select):
		x, y = event.x_root, event.y_root
		dragx0, dragx1 = minmax(drag_start[0], x)
		dragy0, dragy1 = minmax(drag_start[1], y)

		def dragcontains(widget):
			x0 = widget.winfo_rootx()
			y0 = widget.winfo_rooty()
			x1 = x0 + widget.winfo_width()
			y1 = y0 + widget.winfo_height()
			return not (x0 > dragx1 or x1 < dragx0 or y0 > dragy1 or y1 < dragy0)

		# depending on interation over dicts being consistent
		for initial, cf in zip(drag_startstate, widgets.values()):
			if dragcontains(cf):
				cf.checked.set(select)
			else:
				cf.checked.set(initial)

	top.bind('<ButtonPress-1>', mouseDown)
	top.bind('<ButtonPress-3>', mouseDown)
	top.bind('<B1-Motion>', lambda event: drag(event, True))
	top.bind('<B3-Motion>', lambda event: drag(event, False))
	top.bind('<ButtonRelease-1>', mouseUp)
	top.bind('<ButtonRelease-3>', mouseUp)

	# buttons for clearing/filling all checkboxes
	def clear():
		for box in widgets.values():
			box.checked.set(False)
		save()

	def fill():
		for box in widgets.values():
			box.checked.set(True)
		save()

	def mkbutton(parent, name, command):
		button = Tkinter.Button(parent, text=name, command=command)
		button.pack(side='left')

	buttons = Tkinter.Frame(top)
	mkbutton(buttons, "Clear", clear)
	mkbutton(buttons, "Fill", fill)

	# construct bottom buttons for utility scripts
	def sysOff():
		nodes = selected_cfs()
		for name, crazyflie in nodes.items():
			uri = crazyflie["uri"]
			subprocess.call(["ros2 run crazyflie reboot --uri " + uri + " --mode sysoff"], shell=True)

	def reboot():
		nodes = selected_cfs()
		for name, crazyflie in nodes.items():
			uri = crazyflie["uri"]
			print(name)
			subprocess.call(["ros2 run crazyflie reboot --uri " + uri], shell=True)

	def flashSTM():
		nodes = selected_cfs()
		for name, crazyflie in nodes.items():
			uri = crazyflie["uri"]
			print("Flash STM32 FW to {}".format(uri))
			subprocess.call(["ros2 run crazyflie flash --uri " + uri + " --target stm32 --filename " + args.stm32Fw], shell=True)

	def flashNRF():
		nodes = selected_cfs()
		for name, crazyflie in nodes.items():
			uri = crazyflie["uri"]
			print("Flash NRF51 FW to {}".format(uri))
			subprocess.call(["ros2 run crazyflie flash --uri " + uri + " --target nrf51 --filename " + args.nrf51Fw], shell=True)

	def checkBattery():
		# reset color
		for id, w in widgets.items():
			w.batteryLabel.config(foreground='#999999')

		# query each CF
		nodes = selected_cfs()
		for name, crazyflie in nodes.items():
			uri = crazyflie["uri"]
			cfType = crazyflie["type"]
			bigQuad = cfTypes[cfType]["big_quad"]
			
			try:
				if not bigQuad:
					voltage = subprocess.check_output(["ros2 run crazyflie battery --uri " + uri], shell=True)
				else:
					voltage = subprocess.check_output(["ros2 run crazyflie battery --uri " + uri + " --external 1"], shell=True)
			except subprocess.CalledProcessError:
				voltage = None  # CF not available

			color = '#000000'
			if voltage is not None:
				voltage = float(voltage)
				if voltage < cfTypes[cfType]["battery"]["voltage_warning"]:
					color = '#FF8800'
				if voltage < cfTypes[cfType]["battery"]["voltage_critical"]:
					color = '#FF0000'
				widgetText = "{:.2f} v".format(voltage)
			else:
				widgetText = "Err"

			widgets[name].batteryLabel.config(foreground=color, text=widgetText)

	# def checkVersion():
	# 	for id, w in widgets.items():
	# 		w.versionLabel.config(foreground='#999999')
	# 	proc = subprocess.Popen(
	# 		['python3', SCRIPTDIR + 'version.py'], stdout=subprocess.PIPE)
	# 	versions = dict()
	# 	versionsCount = dict()
	# 	versionForMost = None
	# 	versionForMostCount = 0
	# 	for line in iter(proc.stdout.readline, ''):
	# 		print(line)
	# 		match = re.search("(\d+): ([0-9a-fA-F]+),(\d),([0-9a-fA-F]+)", line)
	# 		if match:
	# 			addr = int(match.group(1))
	# 			v1 = match.group(2)
	# 			modified = int(match.group(3)) == 1
	# 			v2 = match.group(4)
	# 			v = (v1,v2)
	# 			versions[addr] = v
	# 			if v in versionsCount:
	# 				versionsCount[v] += 1
	# 			else:
	# 				versionsCount[v] = 1
	# 			if versionsCount[v] > versionForMostCount:
	# 				versionForMostCount = versionsCount[v]
	# 				versionForMost = v
	# 	for addr, v in versions.items():
	# 		color = '#000000'
	# 		if v != versionForMost:
	# 			color = '#FF0000'
	# 		widgets[addr].versionLabel.config(foreground=color, text=str(v[0])[0:3] + "," + str(v[1])[0:3])

	scriptButtons = Tkinter.Frame(top)
	mkbutton(scriptButtons, "battery", checkBattery)
	# currently not supported
	# mkbutton(scriptButtons, "version", checkVersion)
	mkbutton(scriptButtons, "sysOff", sysOff)
	mkbutton(scriptButtons, "reboot", reboot)
	# mkbutton(scriptButtons, "flash (STM)", flashSTM)
	# mkbutton(scriptButtons, "flash (NRF)", flashNRF)

	# start background threads
	def checkBatteryLoop():
		while True:
			# rely on GIL
			checkBattery()
			time.sleep(10.0) # seconds
	# checkBatteryThread = threading.Thread(target=checkBatteryLoop)
	# checkBatteryThread.daemon = True # so it exits when the main thread exit
	# checkBatteryThread.start()

	# place the widgets in the window and start
	buttons.pack()
	frame.pack(padx=10, pady=10)
	scriptButtons.pack()
	top.mainloop()
