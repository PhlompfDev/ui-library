-- Plorch UI library (bundled from Library.luau + spr.luau by build.py; do not edit).
--!nonstrict
-- Plorch: a dark ember script-hub UI library.
--
--   local Plorch = loadstring(game:HttpGet(URL))()
--   local window = Plorch.new({ title = "Plorch", subtitle = "Bloxburg" })
--   local page = window:addPage({ name = "Autofarm", icon = "Replay" })
--   local section = page:addSection("Jobs")
--   section:addToggle({ title = "Pizza Delivery", callback = function(on) end })
--
-- The look and motion are authored in Roblox Studio with Plopacle; this file rebuilds that
-- UI in code. See docs.md for the full API.

local spr = (function()
--!strict
--!native
---------------------------------------------------------------------
-- spr - Spring-driven motion library
--
-- Copyright (c) 2024 Fractality. All rights reserved.
-- Released under the MIT license.
--
-- Docs & license can be found at https://github.com/Fraktality/spr
--
-- API Summary:
--
-- spr.target(
--    Instance obj,
--    number dampingRatio,
--    number undampedFrequency,
--    dict<string, Variant> targetProperties)
--
--    Animates the given properties towardes the target values,
--    given damping ratio and undamped frequency.
--
--
-- spr.stop(
--    Instance obj[,
--    string property])
--
--    Stops the specified property on an Instance from animating.
--    If no property is specified, all properties of the Instance
--    will stop animating.
--
-- Visualizer: https://www.desmos.com/calculator/rzvw27ljh9
---------------------------------------------------------------------

local STRICT_RUNTIME_TYPES = true -- assert on parameter and property type mismatch
local SLEEP_OFFSET_SQ_LIMIT = (1/3840)^2 -- square of the offset sleep limit
local SLEEP_VELOCITY_SQ_LIMIT = 1e-2^2 -- square of the velocity sleep limit
local SLEEP_ROTATION_DIFF = math.rad(0.01) -- rad
local SLEEP_ROTATION_VELOCITY = math.rad(0.1) -- rad/s
local EPS = 1e-5 -- epsilon for stability checks around pathological frequency/damping values

local RunService: RunService = game:GetService("RunService")

local pi = math.pi
local exp = math.exp
local sin = math.sin
local cos = math.cos
local min = math.min
local max = math.max
local sqrt = math.sqrt
local atan2 = math.atan2
local round = math.round

local function magnitudeSq(vec: {number})
	local out = 0
	for _, v in vec do
		out += v^2
	end
	return out
end

local function distanceSq(vec0: {number}, vec1: {number})
	local out = 0
	for i0, v0 in vec0 do
		out += (vec1[i0] - v0)^2
	end
	return out
end

type TypeMetadata<T> = {
	springType: (dampingRatio: number, frequency: number, pos: number, typedat: TypeMetadata<T>, rawTarget: T) -> LinearSpring<T>,
	toIntermediate: (T) -> {number},
	fromIntermediate: ({number}) -> T,
}

-- Spring for an array of linear values
local LinearSpring = {}

type LinearSpring<T> = typeof(setmetatable({} :: {
	d: number,
	f: number,
	g: {number},
	p: {number},
	v: {number},
	typedat: TypeMetadata<T>,
	rawTarget: T,
}, LinearSpring))

do
	LinearSpring.__index = LinearSpring

	function LinearSpring.new<T>(dampingRatio: number, frequency: number, pos: T, rawGoal: T, typedat)
		local linearPos = typedat.toIntermediate(pos)
		return setmetatable(
			{
				d = dampingRatio,
				f = frequency,
				g = linearPos,
				p = linearPos,
				v = table.create(#linearPos, 0),
				typedat = typedat,
				rawGoal = rawGoal
			},
			LinearSpring
		)
	end

	function LinearSpring.setGoal<T>(self, goal: T)
		self.rawGoal = goal
		self.g = self.typedat.toIntermediate(goal)
	end

	function LinearSpring.setDampingRatio<T>(self: LinearSpring<T>, dampingRatio: number)
		self.d = dampingRatio
	end

	function LinearSpring.setFrequency<T>(self: LinearSpring<T>, frequency: number)
		self.f = frequency
	end

	function LinearSpring.canSleep<T>(self)
		if magnitudeSq(self.v) > SLEEP_VELOCITY_SQ_LIMIT then
			return false
		end

		if distanceSq(self.p, self.g) > SLEEP_OFFSET_SQ_LIMIT then
			return false
		end

		return true
	end

	function LinearSpring.step<T>(self: LinearSpring<T>, dt: number)
		-- Advance the spring simulation by dt seconds.
		-- Take the damped harmonic oscillator ODE:
		--    f^2*(X[t] - g) + 2*d*f*X'[t] + X''[t] = 0
		-- Where X[t] is position at time t, g is target position,
		-- f is undamped angular frequency, and d is damping ratio.
		-- Apply constant initial conditions:
		--    X[0] = p0
		--    X'[0] = v0
		-- Solve the IVP to get analytic expressions for X[t] and X'[t].
		-- The solution takes one of three forms for 0<=d<1, d=1, and d>1

		local d = self.d
		local f = self.f*(2*pi) -- Hz -> Rad/s
		local g = self.g
		local p = self.p
		local v = self.v

		if d == 1 then -- critically damped
			local q = exp(-f*dt)
			local w = dt*q

			local c0 = q + w*f
			local c2 = q - w*f
			local c3 = w*f*f

			for idx = 1, #p do
				local o = p[idx] - g[idx]
				p[idx] = o*c0 + v[idx]*w + g[idx]
				v[idx] = v[idx]*c2 - o*c3
			end

		elseif d < 1 then -- underdamped
			local q = exp(-d*f*dt)
			local c = sqrt(1 - d*d)

			local i = cos(dt*f*c)
			local j = sin(dt*f*c)

			-- Damping ratios approaching 1 can cause division by very small numbers.
			-- To mitigate that, group terms around z=j/c and find an approximation for z.
			-- Start with the definition of z:
			--    z = sin(dt*f*c)/c
			-- Substitute a=dt*f:
			--    z = sin(a*c)/c
			-- Take the Maclaurin expansion of z with respect to c:
			--    z = a - (a^3*c^2)/6 + (a^5*c^4)/120 + O(c^6)
			--    z ≈ a - (a^3*c^2)/6 + (a^5*c^4)/120
			-- Rewrite in Horner form:
			--    z ≈ a + ((a*a)*(c*c)*(c*c)/20 - c*c)*(a*a*a)/6

			local z
			if c > EPS then
				z = j/c
			else
				local a = dt*f
				z = a + ((a*a)*(c*c)*(c*c)/20 - c*c)*(a*a*a)/6
			end

			-- Frequencies approaching 0 present a similar problem.
			-- We want an approximation for y as f approaches 0, where:
			--    y = sin(dt*f*c)/(f*c)
			-- Substitute b=dt*c:
			--    y = sin(b*c)/b
			-- Now reapply the process from z.

			local y
			if f*c > EPS then
				y = j/(f*c)
			else
				local b = f*c
				y = dt + ((dt*dt)*(b*b)*(b*b)/20 - b*b)*(dt*dt*dt)/6
			end

			for idx = 1, #p do
				local o = p[idx] - g[idx]
				p[idx] = (o*(i + z*d) + v[idx]*y)*q + g[idx]
				v[idx] = (v[idx]*(i - z*d) - o*(z*f))*q
			end

		else -- overdamped
			local c = sqrt(d*d - 1)

			local r1 = -f*(d + c)
			local r2 = -f*(d - c)

			local ec1 = exp(r1*dt)
			local ec2 = exp(r2*dt)

			for idx = 1, #p do
				local o = p[idx] - g[idx]
				local co2 = (v[idx] - o*r1)/(2*f*c)
				local co1 = ec1*(o - co2)

				p[idx] = co1 + co2*ec2 + g[idx]
				v[idx] = co1*r1 + co2*ec2*r2
			end
		end

		return self.typedat.fromIntermediate(self.p)
	end
end

local RotationSpring = {} 

type RotationSpring = typeof(setmetatable({} :: {
	d: number,
	f: number,
	g: CFrame,
	p: CFrame,
	v: Vector3,
}, RotationSpring))

do
	RotationSpring.__index = RotationSpring

	function RotationSpring.new(d: number, f: number, p: CFrame, g: CFrame)
		return setmetatable(
			{
				d = d,
				f = f,
				g = g:Orthonormalize(),
				p = p:Orthonormalize(),
				v = Vector3.zero
			},
			RotationSpring
		)
	end

	function RotationSpring.setGoal(self: RotationSpring, value: CFrame)
		self.g = value:Orthonormalize()
	end

	function RotationSpring.setDampingRatio(self: RotationSpring, dampingRatio: number)
		self.d = dampingRatio
	end

	function RotationSpring.setFrequency(self: RotationSpring, frequency: number)
		self.f = frequency
	end

	-- evaluate dot products in high precision
	local function dot(v0: Vector3, v1: Vector3)
		return v0.X*v1.X + v0.Y*v1.Y + v0.Z*v1.Z
	end

	local function areRotationsClose(c0: CFrame, c1: CFrame)
		local rx = dot(c0.XVector, c1.XVector)
		local ry = dot(c0.YVector, c1.YVector)
		local rz = dot(c0.ZVector, c1.ZVector)
		local trace = rx + ry + rz
		return trace > 1 + 2*cos(SLEEP_ROTATION_DIFF)
	end
	
	local function angleDiff(c0: CFrame, c1: CFrame)
		local x = dot(c0.XVector, c1.XVector)
		local y = dot(c0.YVector, c1.YVector)
		local z = dot(c0.ZVector, c1.ZVector)
		local w = x + y + z - 1
		return atan2(sqrt(max(0, 1 - w*w*0.25)), w*0.5)
	end
	
	-- gives approx. 21% accuracy improvement over CFrame.fromAxisAngle near poles
	local function fromAxisAngle(axis: Vector3, angle: number)
		local c = cos(angle)
		local s = sin(angle)
		local x, y, z = axis.X, axis.Y, axis.Z

		local mxy = x*y*(1 - c)
		local myz = y*z*(1 - c)
		local mzx = z*x*(1 - c)

		local rx = Vector3.new(x*x*(1 - c) + c, mxy + z*s, mzx - y*s)
		local ry = Vector3.new(mxy - z*s, y*y*(1 - c) + c, myz + x*s)
		local rz = Vector3.new(mzx + y*s, myz - x*s, z*z*(1 - c) + c)

		return CFrame.fromMatrix(Vector3.zero, rx, ry, rz):Orthonormalize()
	end

	local function rotateAxis(r0: Vector3, c1: CFrame)
		local c0 = CFrame.identity
		local mag = r0.Magnitude
		if mag > 1e-6 then
			c0 = fromAxisAngle(r0.Unit, mag)
		end
		return c0 * c1
	end

	-- axis*angle difference between two cframes
	local function axisAngleDiff(c0: CFrame, c1: CFrame)
		-- use native axis (stable enough)
		local axis = (c0*c1:Inverse()):ToAxisAngle()
		
		-- use full-precision angle calculation to minimize truncation
		local angle = angleDiff(c0, c1)
		return axis.Unit*angle
	end

	function RotationSpring.canSleep(self: RotationSpring)
		local sleepP = areRotationsClose(self.p, self.g)
		local sleepV = self.v.Magnitude < SLEEP_ROTATION_VELOCITY
		return sleepP and sleepV
	end

	function RotationSpring.step(self: RotationSpring, dt: number): CFrame
		local d = self.d
		local f = self.f*(2*pi)
		local g = self.g
		local p0 = self.p
		local v0 = self.v

		local offset = axisAngleDiff(p0, g)
		local decay = exp(-d*f*dt)

		local pt: CFrame
		local vt: Vector3

		if d == 1 then -- critically damped
			pt = rotateAxis((offset*(1 + f*dt) + v0*dt)*decay, g)
			vt = (v0*(1 - dt*f) - offset*(dt*f*f))*decay

		elseif d < 1 then -- underdamped
			local c = sqrt(1 - d*d)

			local i = cos(dt*f*c)
			local j = sin(dt*f*c)

			local y = j/(f*c)
			local z = j/c

			pt = rotateAxis((offset*(i + z*d) + v0*y)*decay, g)
			vt = (v0*(i - z*d) - offset*(z*f))*decay

		else -- overdamped
			local c = sqrt(d*d - 1)

			local r1 = -f*(d + c)
			local r2 = -f*(d - c)

			local co2 = (v0 - offset*r1)/(2*f*c)
			local co1 = offset - co2

			local e1 = co1*exp(r1*dt)
			local e2 = co2*exp(r2*dt)

			pt = rotateAxis(e1 + e2, g)
			vt = e1*r1 + e2*r2
		end

		self.p = pt
		self.v = vt

		return pt
	end
end

-- Defined early to be used by CFrameSpring
local typeMetadata_Vector3 = {
	springType = LinearSpring.new,

	toIntermediate = function(value)
		return {value.X, value.Y, value.Z}
	end,

	fromIntermediate = function(value: {number})
		return Vector3.new(value[1], value[2], value[3])
	end,
}

-- Encapsulates a CFrame - Separates translation from rotation
local CFrameSpring = {}
do
	CFrameSpring.__index = CFrameSpring

	function CFrameSpring.new(
		dampingRatio: number,
		frequency: number,
		valueCurrent: CFrame,
		valueGoal: CFrame,
		_: any
	)
		return setmetatable(
			{
				rawGoal = valueGoal,
				_position = LinearSpring.new(dampingRatio, frequency, valueCurrent.Position, valueGoal.Position, typeMetadata_Vector3),
				_rotation = RotationSpring.new(dampingRatio, frequency, valueCurrent.Rotation, valueGoal.Rotation)
			},
			CFrameSpring
		)
	end

	function CFrameSpring:setGoal(value: CFrame)
		self.rawGoal = value
		self._position:setGoal(value.Position)
		self._rotation:setGoal(value.Rotation)
	end

	function CFrameSpring:setDampingRatio(value: number)
		self._position.d = value
		self._rotation.d = value
	end

	function CFrameSpring:setFrequency(value: number)
		self._position.f = value
		self._rotation.f = value
	end

	function CFrameSpring:canSleep()
		return self._position:canSleep() and self._rotation:canSleep()
	end

	function CFrameSpring:step(dt): CFrame
		local p: Vector3 = self._position:step(dt)
		local r: CFrame = self._rotation:step(dt)
		return r + p
	end
end

-- Color conversions
local rgbToLuv
local luvToRgb
do
	local function inverseGammaCorrectD65(c)
		return c < 0.0404482362771076 and c/12.92 or 0.87941546140213*(c + 0.055)^2.4
	end

	local function gammaCorrectD65(c)
		return c < 3.1306684425e-3 and 12.92*c or 1.055*c^(1/2.4) - 0.055
	end

	function rgbToLuv(value: Color3): {number}
		-- convert RGB to a variant of cieluv space
		local r, g, b = value.R, value.G, value.B

		-- D65 sRGB inverse gamma correction
		r = inverseGammaCorrectD65(r)
		g = inverseGammaCorrectD65(g)
		b = inverseGammaCorrectD65(b)

		-- sRGB -> xyz
		local x = 0.9257063972951867*r - 0.8333736323779866*g - 0.09209820666085898*b
		local y = 0.2125862307855956*r + 0.71517030370341085*g + 0.0722004986433362*b
		local z = 3.6590806972265883*r + 11.4426895800574232*g + 4.1149915024264843*b

		-- xyz -> scaled cieluv
		local l = y > 0.008856451679035631 and 116*y^(1/3) - 16 or 903.296296296296*y

		local u, v
		if z > 1e-14 then
			u = l*x/z
			v = l*(9*y/z - 0.46832)
		else
			u = -0.19783*l
			v = -0.46832*l
		end

		return {l, u, v}
	end

	function luvToRgb(value: {number}): Color3
		-- convert back from modified cieluv to rgb space
		local l = value[1]
		if l < 0.0197955 then
			return Color3.new(0, 0, 0)
		end
		local u = value[2]/l + 0.19783
		local v = value[3]/l + 0.46832

		-- cieluv -> xyz
		local y = (l + 16)/116
		y = y > 0.206896551724137931 and y*y*y or 0.12841854934601665*y - 0.01771290335807126
		local x = y*u/v
		local z = y*((3 - 0.75*u)/v - 5)

		-- xyz -> D65 sRGB
		local r =  7.2914074*x - 1.5372080*y - 0.4986286*z
		local g = -2.1800940*x + 1.8757561*y + 0.0415175*z
		local b =  0.1253477*x - 0.2040211*y + 1.0569959*z

		-- clamp minimum sRGB component
		if r < 0 and r < g and r < b then
			r, g, b = 0, g - r, b - r
		elseif g < 0 and g < b then
			r, g, b = r - g, 0, b - g
		elseif b < 0 then
			r, g, b = r - b, g - b, 0
		end

		-- gamma correction from D65
		-- clamp to avoid undesirable overflow wrapping behavior on certain properties (e.g. BasePart.Color)
		return Color3.new(
			min(gammaCorrectD65(r), 1),
			min(gammaCorrectD65(g), 1),
			min(gammaCorrectD65(b), 1)
		)
	end
end

-- Type definitions
-- Transforms Roblox types into intermediate types, converting
-- between spaces as necessary to preserve perceptual linearity
local typeMetadata = {
	boolean = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value and 1 or 0}
		end,

		fromIntermediate = function(value)
			return value[1] >= 0.5
		end,
	},

	number = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value}
		end,

		fromIntermediate = function(value)
			return value[1]
		end,
	},

	NumberRange = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value.Min, value.Max}
		end,

		fromIntermediate = function(value)
			return NumberRange.new(value[1], value[2])
		end,
	},

	UDim = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value.Scale, value.Offset}
		end,

		fromIntermediate = function(value: {number})
			return UDim.new(value[1], round(value[2]))
		end,
	},

	UDim2 = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			local x = value.X
			local y = value.Y
			return {x.Scale, x.Offset, y.Scale, y.Offset}
		end,

		fromIntermediate = function(value: {number})
			return UDim2.new(value[1], round(value[2]), value[3], round(value[4]))
		end,
	},

	Vector2 = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			return {value.X, value.Y}
		end,

		fromIntermediate = function(value: {number})
			return Vector2.new(value[1], value[2])
		end,
	},

	Vector3 = typeMetadata_Vector3,

	Color3 = {
		springType = LinearSpring.new,
		toIntermediate = rgbToLuv,
		fromIntermediate = luvToRgb,
	},

	-- Only interpolates start and end keypoints
	ColorSequence = {
		springType = LinearSpring.new,

		toIntermediate = function(value)
			local keypoints = value.Keypoints

			local luv0 = rgbToLuv(keypoints[1].Value)
			local luv1 = rgbToLuv(keypoints[#keypoints].Value)

			return {
				luv0[1], luv0[2], luv0[3],
				luv1[1], luv1[2], luv1[3],
			}
		end,

		fromIntermediate = function(value: {})
			return ColorSequence.new(
				luvToRgb{value[1], value[2], value[3]},
				luvToRgb{value[4], value[5], value[6]}
			)
		end,
	},

	CFrame = {
		springType = CFrameSpring.new,
		toIntermediate = error, -- custom (CFrameSpring)
		fromIntermediate = error, -- custom (CFrameSpring)
	}
}

type PropertyOverride = {
	[string]: {
		class: string,
		get: (any)->(),
		set: (any, any)->(),
	}
}

local PSEUDO_PROPERTIES: PropertyOverride = {
	Pivot = {
		class = "PVInstance",
		get = function(inst: PVInstance)
			return inst:GetPivot()
		end,
		set = function(inst: PVInstance, value: CFrame)
			inst:PivotTo(value)
		end
	},
	Scale = {
		class = "Model",
		get = function(inst: Model)
			return inst:GetScale()
		end,
		set = function(inst: Model, value: number)
			local FLOAT_MANTISSA_MIN = 1.402e-45
			local FLOAT_MANTISSA_MAX = 2^24
			value = math.clamp(value, FLOAT_MANTISSA_MIN, FLOAT_MANTISSA_MAX)
			inst:ScaleTo(value)
		end
	}
}

local function getProperty(instance: Instance, property: string): any
	local override = PSEUDO_PROPERTIES[property]
	if override and instance:IsA(override.class) then
		return override.get(instance)
	else
		return (instance :: any)[property]
	end
end

local function setProperty(instance: Instance, property: string, value: unknown)
	local override = PSEUDO_PROPERTIES[property]
	if override and instance:IsA(override.class) then
		override.set(instance, value)
	else
		(instance :: any)[property] = value
	end
end

-- Frame loop
local springStates_other: {[Instance]: {[string]: any}} = {} -- {[instance] = {[property] = spring}
local springStates_render: {[Instance]: {[string]: any}} = {} -- {[instance] = {[property] = spring}
local completedCallbacks: {[Instance]: {()->()}} = {}

local function processSprings(springStates: typeof(springStates_other), dt: number)
	for instance, state in springStates do
		for propName, spring in state do
			if spring:canSleep() then
				state[propName] = nil
				setProperty(instance, propName, spring.rawGoal)
			else
				setProperty(instance, propName, spring:step(dt))
			end
		end

		if not next(state) then
			springStates[instance] = nil

			-- trigger completed callbacks when all properties finish animating
			local callbackList = completedCallbacks[instance]
			if callbackList then
				-- flush callback list before we run any callbacks in case
				-- one of the callbacks recursively adds another callback
				completedCallbacks[instance] = nil

				for _, callback in callbackList do
					task.spawn(callback)
				end
			end
		end
	end
end

-- Step-source selection by run context. In a running game the simulation clock
-- drives springs (PreSimulation for most props, PostSimulation for Cameras) —
-- original behavior, unchanged. In Studio edit mode (plugins / edit-mode tooling)
-- the simulation is paused, so PreSimulation/PostSimulation never fire and springs
-- would sit registered-but-unstepped; RenderStepped is the one loop that runs there,
-- so drive both buckets off it. This edit is mirrored byte-for-byte between
-- server/assets/spr.luau and plugin/src/vendor/spr.luau (the drift test enforces it).
-- IsEdit() requires the "Plugin" security capability, which this module lacks when
-- required by an ordinary game LocalScript (this asset ships into real games via
-- install_motion_deps) — pcall-guard the check; "can't even ask" can only mean a
-- non-plugin context, so it falls back to the simulation-event branch, the correct
-- default for a live game.
local okIsEdit, isEditMode = pcall(function()
	return RunService:IsEdit()
end)
if okIsEdit and isEditMode then
	RunService.RenderStepped:Connect(function(dt)
		processSprings(springStates_other, dt)
		processSprings(springStates_render, dt)
	end)
else
	RunService.PreSimulation:Connect(function(dt)
		processSprings(springStates_other, dt)
	end)

	RunService.PostSimulation:Connect(function(dt)
		processSprings(springStates_render, dt)
	end)
end

local function assertType(argNum: number, fnName: string, expectedType: string, value: unknown)
	if not expectedType:find(typeof(value)) then
		error(`bad argument #{argNum} to {fnName} ({expectedType} expected, got {typeof(value)})`, 3)
	end
end

-- API
local spr = {}

function spr.target(instance: Instance, dampingRatio: number, frequency: number, properties: {[string]: any})
	if STRICT_RUNTIME_TYPES then
		assertType(1, "spr.target", "Instance", instance)
		assertType(2, "spr.target", "number", dampingRatio)
		assertType(3, "spr.target", "number", frequency)
		assertType(4, "spr.target", "table", properties)
	end

	if dampingRatio ~= dampingRatio or dampingRatio < 0 then
		error(("expected damping ratio >= 0; got %.2f"):format(dampingRatio), 2)
	end

	if frequency ~= frequency or frequency < 0 then
		error(("expected undamped frequency >= 0; got %.2f"):format(frequency), 2)
	end

	local targetRecord = (if instance:IsA("Camera") then springStates_render else springStates_other) :: {[Instance]: {[string]: any}}

	local state = targetRecord[instance]
	if not state then
		state = {}
		targetRecord[instance] = state
	end

	for propName, propTarget in properties do
		local propValue = getProperty(instance, propName)

		if STRICT_RUNTIME_TYPES and typeof(propTarget) ~= typeof(propValue) then
			error(`bad property {propName} to spr.target ({typeof(propValue)} expected, got {typeof(propTarget)})`, 2)
		end

		-- Special case infinite frequency for an instantaneous change
		if frequency == math.huge then
			setProperty(instance, propName, propTarget)
			state[propName] = nil
			continue
		end

		local spring = state[propName]
		if not spring then
			local md = typeMetadata[typeof(propTarget)]
			if not md then
				error("unsupported type: " .. typeof(propTarget), 2)
			end

			spring = md.springType(dampingRatio, frequency, propValue, propTarget, md)
			state[propName] = spring
		end

		spring:setGoal(propTarget)
		spring:setDampingRatio(dampingRatio)
		spring:setFrequency(frequency)
	end

	if not next(state) then
		targetRecord[instance] = nil
	end
end

function spr.stop(instance: Instance, property: string?)
	if STRICT_RUNTIME_TYPES then
		assertType(1, "spr.stop", "Instance", instance)
		assertType(2, "spr.stop", "string|nil", property)
	end

	if property then
		local state = springStates_other[instance] or springStates_render[instance]
		if state then
			state[property] = nil
		end
	else
		springStates_other[instance] = nil
		springStates_render[instance] = nil
	end
end

function spr.completed(instance: Instance, callback: ()->())
	if STRICT_RUNTIME_TYPES then
		assertType(1, "spr.completed", "Instance", instance)
		assertType(2, "spr.completed", "function", callback)
	end

	local callbackList = completedCallbacks[instance]
	if callbackList then
		table.insert(callbackList, callback)
	else
		completedCallbacks[instance] = {callback}
	end
end

return table.freeze(spr)
end)()

local HttpService = game:GetService("HttpService")
local Players = game:GetService("Players")
local RunService = game:GetService("RunService")
local UserInputService = game:GetService("UserInputService")
local GuiService = game:GetService("GuiService")

local Plorch = { version = "0.1.0" }

---------------------------------------------------------------------------------------------
-- Theme, motion numbers and icons

-- Colour tokens. Pass `theme = { accent = Color3.new(...) }` to Plorch.new to override any.
local THEME = {
	sidebar = "#110F0D",
	bg = "#171513",
	card = "#201C1A",
	menu = "#221E1C",
	field = "#24201E",
	fieldHover = "#2E2926",
	control = "#2F2A27",
	optionHover = "#342E2A",
	tabActive = "#3B2213",
	border = "#4C433D",
	knobOff = "#7A6F67",
	textDim = "#A39A92",
	text = "#F2ECE6",
	accent = "#FF7E33",
	accent2 = "#FFBF40",
	white = "#FFFFFF",
	good = "#77C777",
	warn = "#FFBF40",
	bad = "#E6625A",
}

-- hex -> token, so the hex literals in TUNING follow a theme override.
local HEX_TOKENS = {}
for token, hex in pairs(THEME) do
	if token ~= "warn" then
		HEX_TOKENS[hex] = token
	end
end

local CLOSE_HIDE_DELAY = 0.3 -- seconds the close spring gets before the window stops rendering
local MAX_RESULTS = 30
local TOAST_DURATION = 4
local TOAST_EXIT_TIME = 0.3
local OPTION_HEIGHT = 32
local OPTION_STAGGER = 0.025 -- seconds between each option fading in
local MENU_COLLAPSED_HEIGHT = 10 -- the menu unrolls from, and rolls back to, this height
local MENU_CLOSE_TIME = 0.16 -- seconds the close gets before the menu stops rendering
local OPTION_SELECTED_TRANSPARENCY = 0.5 -- resting fill of the current choice (hover is 0)
local MENU_MAX_VISIBLE = 6 -- options a menu shows at once; the rest scroll
local MENU_SEARCH_HEIGHT = 32 -- a menu's search field plus the gap above it
local CONFIG_FORMAT = 1 -- version written into every config payload
local SECTION_GAP = 16 -- space under each section; part of the section so it collapses with it
local AUTOSAVE_DELAY = 0.5
local REFERENCE_HEIGHT = 1080 -- the screen height the window's pixel sizes were drawn for
local MAX_FIT = 2 -- the window never grows past twice its drawn size
local MIN_FIT = 0.45 -- nor shrinks below this, however small the screen
local SCREEN_MARGIN = 24 -- pixels kept free around the window when it has to shrink
local MOBILE_BUTTON_SIZE = 46
local TAP_SLOP = 8 -- pixels a touch may wander and still count as a tap, not a drag

-- Spring numbers and animated goals, keyed by element kind. Kept identical to the TUNING
-- table of the Studio original so `verify_export` can diff the two.
local TUNING = {
	["window"] = {
		open = { damping = 1, frequency = 4.5, from = { GroupTransparency = 1 } },
		close = { damping = 1, frequency = 6, goal = { GroupTransparency = 1 } },
	},
	["window-scale"] = {
		open = { damping = 0.7, frequency = 4, from = { Scale = 0.92 } },
		close = { damping = 1, frequency = 6, goal = { Scale = 0.95 } },
	},
	["page-host"] = {
		enter = { damping = 1, frequency = 5, from = { GroupTransparency = 1, PositionOffsetY = 12 } },
	},
	["tab"] = {
		active = { damping = 1, frequency = 5, goal = { BackgroundTransparency = 0 } },
		inactive = { damping = 1, frequency = 5, goal = { BackgroundTransparency = 1 } },
		hover = { damping = 1, frequency = 6, goal = { BackgroundTransparency = 0.5 } },
	},
	["tab-indicator"] = {
		active = { damping = 0.6, frequency = 4, goal = { Size = "UDim2(0,3,0,16)" } },
		inactive = { damping = 1, frequency = 6, goal = { Size = "UDim2(0,3,0,0)" } },
	},
	["tab-label"] = {
		active = { damping = 1, frequency = 5, goal = { TextColor3 = "#F2ECE6" } },
		inactive = { damping = 1, frequency = 5, goal = { TextColor3 = "#A39A92" } },
	},
	["tab-icon"] = {
		active = { damping = 1, frequency = 5, goal = { GroupColor3 = "#FF7E33" } },
		inactive = { damping = 1, frequency = 5, goal = { GroupColor3 = "#A39A92" } },
	},
	["toggle-track"] = {
		on = { damping = 1, frequency = 5, goal = { BackgroundColor3 = "#FF7E33" } },
		off = { damping = 1, frequency = 5, goal = { BackgroundColor3 = "#2F2A27" } },
	},
	["toggle-knob"] = {
		on = { damping = 0.65, frequency = 4.5, goal = { Position = "UDim2(0,29,0.5,0)", BackgroundColor3 = "#171513" } },
		off = { damping = 0.65, frequency = 4.5, goal = { Position = "UDim2(0,11,0.5,0)", BackgroundColor3 = "#7A6F67" } },
		press = { damping = 1, frequency = 8, goal = { Size = "UDim2(0,20,0,16)" } },
	},
	["slider-fill"] = {
		drag = { damping = 1, frequency = 9 },
	},
	["slider-knob"] = {
		drag = { damping = 1, frequency = 9 },
		hover = { damping = 0.6, frequency = 5, goal = { Size = "UDim2(0,10,0,22)" } },
	},
	["button"] = {
		hover = { damping = 1, frequency = 6, goal = { BackgroundColor3 = "#FF7E33", TextColor3 = "#171513" } },
		press = { damping = 1, frequency = 8, goal = { Size = "UDim2(0,76,0,28)" } },
	},
	["dropdown"] = {
		hover = { damping = 1, frequency = 6, goal = { BackgroundColor3 = "#2E2926" } },
	},
	["dropdown-stroke"] = {
		open = { damping = 1, frequency = 5, goal = { Color = "#FF7E33", Transparency = 0.1 } },
	},
	["dropdown-chevron"] = {
		open = { damping = 0.6, frequency = 4.5, goal = { Rotation = 180 } },
	},
	["dropdown-value"] = {
		change = { damping = 0.8, frequency = 5, from = { TextTransparency = 1, PositionOffsetY = 8 } },
	},
	["dropdown-menu"] = {
		open = { damping = 1, frequency = 7, from = { GroupTransparency = 1 } },
		close = { damping = 1, frequency = 9, goal = { GroupTransparency = 1 } },
		grow = { damping = 0.75, frequency = 5 },
		shrink = { damping = 1, frequency = 8 },
	},
	["option"] = {
		hover = { damping = 1, frequency = 8, goal = { BackgroundTransparency = 0 } },
		enter = { damping = 1, frequency = 6, from = { TextTransparency = 1 } },
	},
	["keybind"] = {
		hover = { damping = 1, frequency = 6, goal = { BackgroundColor3 = "#2E2926" } },
	},
	["keybind-stroke"] = {
		listening = { damping = 1, frequency = 5, goal = { Color = "#FF7E33", Transparency = 0.1 } },
	},
	["field-stroke"] = {
		focus = { damping = 1, frequency = 5, goal = { Color = "#FF7E33", Transparency = 0.1 } },
	},
	["search-bar"] = {
		focus = { damping = 0.8, frequency = 4, goal = { Size = "UDim2(0,280,0,32)" } },
	},
	["search-stroke"] = {
		focus = { damping = 1, frequency = 5, goal = { Color = "#FF7E33", Transparency = 0.1 } },
	},
	["result-jump"] = {
		hover = { damping = 1, frequency = 6, goal = { TextColor3 = "#FF7E33" } },
	},
	["row-title"] = {
		flash = { damping = 1, frequency = 1.2 },
	},
	["toast"] = {
		enter = { damping = 1, frequency = 4.5, from = { GroupTransparency = 1 } },
		exit = { damping = 1, frequency = 6, goal = { GroupTransparency = 1 } },
	},
	["toast-scale"] = {
		enter = { damping = 0.7, frequency = 4, from = { Scale = 0.9 } },
		exit = { damping = 1, frequency = 6, goal = { Scale = 0.95 } },
	},
	-- Library-only elements (not in the Studio original).
	["action"] = {
		hover = { damping = 1, frequency = 6, goal = { BackgroundColor3 = "#FF7E33", TextColor3 = "#171513" } },
	},
	["action-primary"] = {
		hover = { damping = 1, frequency = 6, goal = { BackgroundColor3 = "#FFBF40" } },
	},
	["section"] = {
		show = { damping = 0.85, frequency = 4.5 },
		hide = { damping = 1, frequency = 6 },
	},
}

-- Sprite-sheet glyphs: name -> { asset id, offset x, offset y, cell size }.
local ICONS = {
	["3x3MenuGrid"] = { "6764432408", 150, 50, 50 },
	AccountCircle = { "8445470984", 804, 804, 96 },
	AddCircle = { "8445470984", 604, 504, 96 },
	ArrowForward = { "8445470826", 804, 404, 96 },
	ArrowUpward = { "8445470826", 304, 404, 96 },
	Bookmark = { "8445470392", 204, 704, 96 },
	Check = { "8445471173", 504, 604, 96 },
	CheckBoxOutlineBlank = { "8445519745", 940, 784, 48 },
	Cloud = { "8445470984", 504, 304, 96 },
	Cursor = { "6764432293", 400, 0, 100 },
	Delete = { "8445470559", 604, 104, 96 },
	Edit = { "8445470559", 704, 504, 96 },
	Elipses = { "8445471713", 704, 404, 96 },
	Exit = { "6764432408", 200, 350, 50 },
	ExitRight = { "6764432408", 200, 250, 50 },
	Export = { "6764432408", 200, 200, 50 },
	FastForward = { "6764432408", 200, 150, 50 },
	Fingerprint = { "8445470826", 204, 104, 96 },
	Folder = { "8445470559", 604, 204, 96 },
	Forum = { "8445470559", 104, 304, 96 },
	Fullscreen = { "8445470826", 404, 604, 96 },
	Gear = { "6764432408", 150, 850, 50 },
	HamburgerBar = { "8445471713", 904, 104, 96 },
	Help = { "8445471713", 804, 704, 96 },
	Home = { "6764432408", 150, 800, 50 },
	Info = { "8445471499", 304, 104, 96 },
	InfoOutline = { "8445471499", 204, 304, 96 },
	Information = { "6764432408", 150, 700, 50 },
	Label = { "8445471499", 504, 404, 96 },
	Link = { "8445470392", 104, 404, 96 },
	Lock = { "8445472085", 4, 104, 96 },
	Map = { "8445471713", 304, 4, 96 },
	Mountain = { "8445470017", 404, 864, 96 },
	Receipt = { "8445471499", 204, 604, 96 },
	Replay = { "8445471499", 404, 804, 96 },
	Reply = { "8445471499", 504, 804, 96 },
	Send = { "8445471332", 804, 104, 96 },
	Smartphone = { "8445470392", 304, 204, 96 },
	Store = { "8445470392", 104, 204, 96 },
	SubdirectoryArrowRight = { "8445470392", 4, 704, 96 },
	SwapHoriz = { "8445470189", 704, 104, 96 },
	SwapVert = { "8445470189", 204, 204, 96 },
	Tablet = { "8445470392", 704, 704, 96 },
	ThumbDown = { "8445470189", 304, 604, 96 },
	ThumbUp = { "8445470189", 504, 504, 96 },
	TouchApp = { "8445470392", 504, 504, 96 },
	VerifiedUser = { "8445470189", 904, 404, 96 },
	Visibility = { "8445470392", 604, 4, 96 },
	VisibilityOff = { "8445470189", 604, 804, 96 },
	VolumeDown = { "8445470189", 4, 104, 96 },
	VolumeMute = { "8445470189", 304, 804, 96 },
	VolumeOff = { "8445470392", 604, 604, 96 },
	VolumeUp = { "8445470392", 204, 604, 96 },
	Warning = { "8445470392", 104, 704, 96 },
	Wrench = { "6764432408", 0, 850, 50 },
	ZoomIn = { "8445470189", 904, 504, 96 },
	ZoomOut = { "8445471713", 504, 804, 96 },
}

local KEY_NAMES = {
	RightControl = "RCtrl", LeftControl = "LCtrl", RightShift = "RShift", LeftShift = "LShift",
	RightAlt = "RAlt", LeftAlt = "LAlt", Return = "Enter", Backspace = "Bksp", CapsLock = "Caps",
	PageUp = "PgUp", PageDown = "PgDn", Insert = "Ins", Delete = "Del",
	One = "1", Two = "2", Three = "3", Four = "4", Five = "5",
	Six = "6", Seven = "7", Eight = "8", Nine = "9", Zero = "0",
}

---------------------------------------------------------------------------------------------
-- Small helpers

local FONT_FAMILY = "rbxasset://fonts/families/BuilderSans.json"

local function font(weight)
	return Font.new(FONT_FAMILY, Enum.FontWeight[weight])
end

-- Instance.new with a property table. `Attr` sets attributes, `Parent` is applied last.
local function make(class, props, children)
	local instance = Instance.new(class)
	if instance:IsA("GuiObject") then
		instance.BorderSizePixel = 0
	end
	local parent
	if props then
		for key, value in pairs(props) do
			if key == "Parent" then
				parent = value
			elseif key == "Attr" then
				for name, attribute in pairs(value) do
					instance:SetAttribute(name, attribute)
				end
			else
				instance[key] = value
			end
		end
	end
	if children then
		for _, child in ipairs(children) do
			child.Parent = instance
		end
	end
	if parent then
		instance.Parent = parent
	end
	return instance
end

local function corner(scale, offset)
	return make("UICorner", { CornerRadius = UDim.new(scale, offset) })
end

local function fullCorner()
	return corner(1, 0)
end

local function keyName(code)
	if not code then
		return "None"
	end
	return KEY_NAMES[code.Name] or code.Name
end

local function parseKey(value)
	if typeof(value) == "EnumItem" then
		return value
	end
	if not value or value == "" or value == "None" then
		return nil
	end
	local ok, code = pcall(function()
		return Enum.KeyCode[value]
	end)
	return ok and code or nil
end

local function isModifier(code)
	local name = code.Name
	return name:find("Control") ~= nil or name:find("Shift") ~= nil or name:find("Alt") ~= nil
end

local function isPointerDown(input)
	return input.UserInputType == Enum.UserInputType.MouseButton1 or input.UserInputType == Enum.UserInputType.Touch
end

local function isPointerMove(input)
	return input.UserInputType == Enum.UserInputType.MouseMovement or input.UserInputType == Enum.UserInputType.Touch
end

-- "Pizza Delivery" -> "PizzaDelivery"; used for instance names and config keys.
local function identifier(text)
	local out = tostring(text):gsub("[^%w]+(%w?)", function(first)
		return first:upper()
	end)
	out = out:gsub("^%l", string.upper)
	return out
end

local function formatNumber(value, step)
	if step % 1 == 0 then
		return ("%d"):format(math.floor(value + 0.5))
	end
	local decimals = math.clamp(math.ceil(-math.log10(step)), 1, 4)
	return ("%." .. decimals .. "f"):format(value)
end

-- Runs a user callback on its own thread so an error (or a yield) in it never reaches the UI.
local function fire(callback, ...)
	if callback then
		task.spawn(function(...)
			local ok, message = pcall(callback, ...)
			if not ok then
				warn("[Plorch] callback error: " .. tostring(message))
			end
		end, ...)
	end
end

-- Executors can leave a thread that has just called into the game unable to touch the UI
-- until it next yields. Display setters go through here: try now, retry after a yield.
local function ui(write)
	if not pcall(write) then
		task.defer(pcall, write)
	end
end

-- Executor globals (gethui, writefile, ...) by name; nil in Studio or when unsupported.
local executorGlobal
function executorGlobal(name)
	for level = 1, 0, -1 do
		local okEnv, env = pcall(getfenv, level == 0 and 0 or executorGlobal)
		if not okEnv or type(env) ~= "table" then
			continue
		end
		local value = env[name]
		if value ~= nil then
			return value
		end
		local shared = env.getgenv
		if type(shared) == "function" then
			local ok, globals = pcall(shared)
			if ok and type(globals) == "table" and globals[name] ~= nil then
				return globals[name]
			end
		end
	end
	return nil
end

-- { writefile, readfile, isfile, makefolder?, isfolder?, listfiles?, delfile? } or nil when
-- files are unavailable.
local function fileApi()
	local files = {}
	for _, name in ipairs({ "writefile", "readfile", "isfile", "makefolder", "isfolder", "listfiles", "delfile" }) do
		local found = executorGlobal(name)
		if type(found) == "function" then
			files[name] = found
		end
	end
	if files.writefile and files.readfile and files.isfile then
		return files
	end
	return nil
end

---------------------------------------------------------------------------------------------
-- Motion runtime (twin of the Animation Editor's codec: UDim2, #RRGGBB, number)

local function newMotion(theme)
	local motion = {}

	local function decode(v)
		if type(v) ~= "string" then
			return v
		end
		local a, b, c, d = v:match("^UDim2%(%s*(-?[%d%.]+)%s*,%s*(-?[%d%.]+)%s*,%s*(-?[%d%.]+)%s*,%s*(-?[%d%.]+)%s*%)$")
		if a then
			return UDim2.new(tonumber(a), tonumber(b), tonumber(c), tonumber(d))
		end
		if v:match("^#%x%x%x%x%x%x$") then
			local token = HEX_TOKENS[v]
			return token and theme[token] or Color3.fromHex(v)
		end
		return v
	end

	local function decodeProps(map)
		local out = {}
		for k, v in pairs(map) do
			out[k] = decode(v)
		end
		return out
	end
	motion.decodeProps = decodeProps

	function motion.entry(slug, category)
		local categories = TUNING[slug]
		return categories and categories[category]
	end
	local entry = motion.entry

	-- Resting value of each animated property, captured the first time it is touched.
	-- Weak keys: cloned toasts, options and result cards drop out when they are destroyed.
	local restStore = setmetatable({}, { __mode = "k" })
	local function restOf(node)
		local store = restStore[node]
		if not store then
			store = {}
			restStore[node] = store
		end
		return store
	end
	motion.restOf = restOf

	function motion.remember(node, slug, category)
		local e = entry(slug, category)
		if not (node and e and e.goal) then
			return
		end
		local store = restOf(node)
		for prop in pairs(e.goal) do
			if store[prop] == nil then
				store[prop] = node[prop]
			end
		end
	end

	-- Spring rest -> goal. `instant` writes the goal with no motion (initial state).
	function motion.toGoal(node, slug, category, instant)
		local e = entry(slug, category)
		if not (node and e and e.goal) then
			return
		end
		motion.remember(node, slug, category)
		local goal = decodeProps(e.goal)
		if instant then
			spr.stop(node)
			for prop, value in pairs(goal) do
				node[prop] = value
			end
		else
			spr.target(node, e.damping, e.frequency, goal)
		end
	end

	-- Spring the category's properties back to their resting values.
	function motion.toRest(node, slug, category)
		local e = entry(slug, category)
		if not (node and e and e.goal) then
			return
		end
		local store = restOf(node)
		local rest = {}
		for prop in pairs(e.goal) do
			if store[prop] ~= nil then
				rest[prop] = store[prop]
			end
		end
		spr.target(node, e.damping, e.frequency, rest)
	end

	-- Spring offset start -> rest. With `fromCurrent` the start snap is skipped, so an
	-- interrupted exit turns around from wherever it is.
	function motion.playFrom(node, slug, category, fromCurrent)
		local e = entry(slug, category)
		if not (node and e and e.from) then
			return
		end
		local store = restOf(node)
		local rest, start = {}, {}
		for prop, value in pairs(decodeProps(e.from)) do
			if prop == "PositionOffsetX" or prop == "PositionOffsetY" then
				if store.Position == nil then
					store.Position = node.Position
				end
				rest.Position = store.Position
				local offset = prop == "PositionOffsetX" and UDim2.fromOffset(value, 0) or UDim2.fromOffset(0, value)
				start.Position = (start.Position or store.Position) + offset
			else
				if store[prop] == nil then
					store[prop] = node[prop]
				end
				rest[prop] = store[prop]
				start[prop] = value
			end
		end
		if not fromCurrent then
			spr.stop(node)
			for prop, value in pairs(start) do
				node[prop] = value
			end
		end
		spr.target(node, e.damping, e.frequency, rest)
	end

	-- Hover wiring shared by every control that has a `hover` category.
	function motion.bindHover(node, slug)
		motion.remember(node, slug, "hover")
		node.MouseEnter:Connect(function()
			motion.toGoal(node, slug, "hover")
		end)
		node.MouseLeave:Connect(function()
			motion.toRest(node, slug, "hover")
		end)
	end

	return motion
end

---------------------------------------------------------------------------------------------
-- Static builders: each returns instances with no behaviour attached

local function ember(theme, rotation)
	return make("UIGradient", {
		Name = "Ember",
		Color = ColorSequence.new(theme.accent, theme.accent2),
		Rotation = rotation or 0,
	})
end

-- A tab/title icon. `icon` is a name from ICONS, "Sliders" (drawn), an asset id string,
-- or { image = "rbxassetid://...", rectOffset = Vector2, rectSize = Vector2 }.
local function buildIcon(theme, icon)
	if icon == "Sliders" then
		local group = make("CanvasGroup", {
			Name = "Icon",
			AnchorPoint = Vector2.new(0, 0.5),
			BackgroundTransparency = 1,
			Position = UDim2.new(0, 14, 0.5, 0),
			GroupColor3 = theme.textDim,
			Size = UDim2.fromOffset(16, 16),
			Attr = { PlopacleNode = "tab-icon" },
		})
		for index, y in ipairs({ 3, 8, 13 }) do
			make("Frame", {
				Name = "Bar" .. index,
				Parent = group,
				AnchorPoint = Vector2.new(0, 0.5),
				Position = UDim2.fromOffset(1, y),
				BackgroundColor3 = theme.white,
				Size = UDim2.fromOffset(14, 2),
			}, { fullCorner() })
		end
		for index, x in ipairs({ 11, 5, 9 }) do
			make("Frame", {
				Name = "Knob" .. index,
				Parent = group,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.fromOffset(x, ({ 3, 8, 13 })[index]),
				BackgroundColor3 = theme.white,
				Size = UDim2.fromOffset(4, 4),
			}, { fullCorner() })
		end
		return group
	end

	local image, offset, size
	local known = type(icon) == "string" and ICONS[icon]
	if known then
		image = "rbxassetid://" .. known[1]
		offset = Vector2.new(known[2], known[3])
		size = Vector2.new(known[4], known[4])
	elseif type(icon) == "table" then
		image, offset, size = icon.image, icon.rectOffset, icon.rectSize
	elseif type(icon) == "string" then
		image = icon
	elseif type(icon) == "number" then
		image = "rbxassetid://" .. icon
	end

	local group = make("CanvasGroup", {
		Name = "Icon",
		AnchorPoint = Vector2.new(0, 0.5),
		BackgroundTransparency = 1,
		Position = UDim2.new(0, 12, 0.5, 0),
		GroupColor3 = theme.textDim,
		Size = UDim2.fromOffset(20, 20),
		Attr = { PlopacleNode = "tab-icon" },
	})
	local glyph = make("ImageLabel", {
		Name = "Glyph",
		Parent = group,
		Image = image or "",
		BackgroundTransparency = 1,
		ScaleType = Enum.ScaleType.Fit,
		Size = UDim2.fromScale(1, 1),
	})
	if size then
		glyph.ImageRectSize = size
	end
	if offset then
		glyph.ImageRectOffset = offset
	end
	return group
end

local function buildTab(theme, name, icon, order)
	local button = make("TextButton", {
		Name = "Tab_" .. identifier(name),
		LayoutOrder = order,
		BackgroundTransparency = 1,
		BackgroundColor3 = theme.tabActive,
		AutoButtonColor = false,
		Text = "",
		Size = UDim2.new(1, 0, 0, 36),
		Attr = { Tab = name, PlopacleNode = "tab" },
	}, { corner(0, 6) })
	local indicator = make("Frame", {
		Name = "Indicator",
		Parent = button,
		AnchorPoint = Vector2.new(0, 0.5),
		Position = UDim2.new(0, 0, 0.5, 0),
		BackgroundColor3 = theme.accent,
		Size = UDim2.fromOffset(3, 0),
		Attr = { PlopacleNode = "tab-indicator" },
	}, { fullCorner() })
	local label = make("TextLabel", {
		Name = "Label",
		Parent = button,
		FontFace = font("Medium"),
		TextColor3 = theme.textDim,
		Text = name,
		BackgroundTransparency = 1,
		TextXAlignment = Enum.TextXAlignment.Left,
		Position = UDim2.fromOffset(38, 0),
		TextSize = 17,
		Size = UDim2.new(1, -46, 1, 0),
		Attr = { PlopacleNode = "tab-label" },
	})
	local iconGroup = buildIcon(theme, icon)
	iconGroup.Parent = button
	make("UIGradient", {
		Name = "Fade",
		Parent = button,
		Transparency = NumberSequence.new(0, 0.75),
	})
	return { button = button, indicator = indicator, label = label, icon = iconGroup }
end

local function scroller(theme, name, attributes, listPadding)
	return make("ScrollingFrame", {
		Name = name,
		Visible = false,
		AutomaticCanvasSize = Enum.AutomaticSize.Y,
		ScrollBarThickness = 3,
		BackgroundTransparency = 1,
		ScrollingDirection = Enum.ScrollingDirection.Y,
		CanvasSize = UDim2.new(0, 0, 0, 0),
		ScrollBarImageColor3 = theme.border,
		Size = UDim2.fromScale(1, 1),
		Attr = attributes,
	}, {
		make("UIPadding", {
			PaddingBottom = UDim.new(0, 24),
			PaddingTop = UDim.new(0, 16),
			PaddingLeft = UDim.new(0, 24),
			PaddingRight = UDim.new(0, 24),
		}),
		make("UIListLayout", { Padding = UDim.new(0, listPadding), SortOrder = Enum.SortOrder.LayoutOrder }),
	})
end

-- The rounded, stroked card shared by sections and search results.
local function card(theme, name, order, paddingTop)
	return make("Frame", {
		Name = name,
		LayoutOrder = order,
		BackgroundColor3 = theme.card,
		AutomaticSize = Enum.AutomaticSize.Y,
		Size = UDim2.new(1, 0, 0, 0),
	}, {
		corner(0, 8),
		make("UIStroke", { Color = theme.border, Transparency = 0.65 }),
		make("UIPadding", {
			PaddingBottom = UDim.new(0, 8),
			PaddingTop = UDim.new(0, paddingTop),
			PaddingLeft = UDim.new(0, 16),
			PaddingRight = UDim.new(0, 16),
		}),
		make("UIListLayout", { Padding = UDim.new(0, 4), SortOrder = Enum.SortOrder.LayoutOrder }),
	})
end

-- A thin frame inset 1 px that carries a CanvasGroup's edge stroke, so the stroke fades
-- with the group (a UIStroke on the group itself does not).
local function edge(theme, zIndex, transparency, radius)
	return make("Frame", {
		Name = "Edge",
		AnchorPoint = Vector2.new(0.5, 0.5),
		BackgroundTransparency = 1,
		Position = UDim2.fromScale(0.5, 0.5),
		ZIndex = zIndex,
		Size = UDim2.new(1, -2, 1, -2),
	}, {
		corner(0, radius),
		make("UIStroke", {
			Color = theme.border,
			Transparency = transparency,
			ApplyStrokeMode = Enum.ApplyStrokeMode.Border,
		}),
	})
end

-- Title (+ optional description) on the left of a row, leaving `reserve` px for the control.
local function rowBase(theme, class, id, order, height, title, description, reserve, attributes)
	local row = make(class, {
		Name = "Row_" .. id,
		LayoutOrder = order,
		BackgroundTransparency = 1,
		Size = UDim2.new(1, 0, 0, height),
		Attr = attributes,
	})
	if class == "TextButton" then
		row.AutoButtonColor = false
		row.Text = ""
	end
	local hasDescription = description ~= nil and description ~= ""
	make("TextLabel", {
		Name = "Title",
		Parent = row,
		FontFace = font("Medium"),
		TextColor3 = theme.text,
		Text = title or "",
		BackgroundTransparency = 1,
		TextXAlignment = Enum.TextXAlignment.Left,
		Position = UDim2.fromOffset(0, (hasDescription or height ~= 48) and 6 or 15),
		TextSize = 15,
		Size = UDim2.new(1, -reserve, 0, 18),
	})
	if hasDescription then
		make("TextLabel", {
			Name = "Desc",
			Parent = row,
			FontFace = font("Regular"),
			TextColor3 = theme.textDim,
			Text = description,
			BackgroundTransparency = 1,
			TextXAlignment = Enum.TextXAlignment.Left,
			Position = UDim2.fromOffset(0, 26),
			TextSize = 13,
			Size = UDim2.new(1, -reserve, 0, 16),
		})
	end
	return row
end

-- Everything that exists before any page is added.
local function buildShell(theme, options)
	local refs = {}

	local gui = make("ScreenGui", {
		Name = options.name or options.title or "Plorch",
		DisplayOrder = options.displayOrder or 100,
		ResetOnSpawn = false,
		IgnoreGuiInset = true,
		ZIndexBehavior = Enum.ZIndexBehavior.Sibling,
	})
	refs.gui = gui

	local size = options.size or Vector2.new(760, 500)
	local window = make("CanvasGroup", {
		Name = "Window",
		Parent = gui,
		AnchorPoint = Vector2.new(0.5, 0.5),
		Position = UDim2.fromScale(0.5, 0.5),
		BackgroundColor3 = theme.bg,
		Size = UDim2.fromOffset(size.X, size.Y),
		Attr = { PlopacleNode = "window" },
	}, { corner(0, 10) })
	refs.window = window
	refs.windowScale = make("UIScale", { Name = "Scale", Parent = window, Attr = { PlopacleNode = "window-scale" } })

	-- Sidebar
	local sidebar = make("Frame", {
		Name = "Sidebar",
		Parent = window,
		BackgroundColor3 = theme.sidebar,
		Size = UDim2.new(0, 184, 1, 0),
	})
	refs.sidebar = sidebar

	local brand = make("Frame", {
		Name = "Brand",
		Parent = sidebar,
		Position = UDim2.fromOffset(16, 16),
		BackgroundTransparency = 1,
		Size = UDim2.new(1, -32, 0, 28),
	})
	refs.brand = brand
	local title = options.title or "Plorch"
	refs.logo = make("TextLabel", {
		Name = "Logo",
		Parent = brand,
		BackgroundColor3 = theme.white,
		FontFace = font("ExtraBold"),
		TextColor3 = theme.bg,
		Text = options.logo or title:sub(1, 1):upper(),
		TextSize = 16,
		Size = UDim2.fromOffset(28, 28),
	}, { corner(0, 7), ember(theme, 45) })
	refs.brandTitle = make("TextLabel", {
		Name = "Title",
		Parent = brand,
		FontFace = font("ExtraBold"),
		TextColor3 = theme.text,
		Text = title,
		Position = UDim2.fromOffset(40, 0),
		BackgroundTransparency = 1,
		TextXAlignment = Enum.TextXAlignment.Left,
		TextTruncate = Enum.TextTruncate.AtEnd,
		TextSize = 17,
		Size = UDim2.new(1, -40, 0, 16),
	})
	refs.caption = make("TextLabel", {
		Name = "Caption",
		Parent = brand,
		FontFace = font("Regular"),
		TextColor3 = theme.textDim,
		Text = options.subtitle or "",
		BackgroundTransparency = 1,
		TextXAlignment = Enum.TextXAlignment.Left,
		Position = UDim2.fromOffset(40, 17),
		TextSize = 12,
		Size = UDim2.new(1, -40, 0, 11),
	})

	refs.tabList = make("Frame", {
		Name = "TabList",
		Parent = sidebar,
		Position = UDim2.fromOffset(12, 69),
		BackgroundTransparency = 1,
		Size = UDim2.new(1, -24, 1, -85),
	}, {
		make("UIListLayout", { Padding = UDim.new(0, 4), SortOrder = Enum.SortOrder.LayoutOrder }),
	})
	-- A flexible gap: tabs above it stack from the top, the pinned tab below sits at the bottom.
	make("Frame", {
		Name = "Spacer",
		Parent = refs.tabList,
		LayoutOrder = 99,
		BackgroundTransparency = 1,
		Size = UDim2.new(1, 0, 0, 0),
	}, { make("UIFlexItem", { FlexMode = Enum.UIFlexMode.Fill }) })

	make("Frame", {
		Name = "Wash",
		Parent = sidebar,
		BackgroundColor3 = theme.accent,
		ZIndex = 0,
		Size = UDim2.new(1, 0, 0, 220),
	}, {
		make("UIGradient", { Rotation = 90, Transparency = NumberSequence.new(0.86, 1) }),
	})
	make("Frame", {
		Name = "BrandDivider",
		Parent = sidebar,
		BackgroundTransparency = 0.5,
		Position = UDim2.fromOffset(16, 56),
		BackgroundColor3 = theme.white,
		Size = UDim2.new(1, -32, 0, 1),
	}, {
		make("UIGradient", {
			Color = ColorSequence.new(theme.accent, theme.border),
			Transparency = NumberSequence.new({
				NumberSequenceKeypoint.new(0, 0),
				NumberSequenceKeypoint.new(0.5, 0.2),
				NumberSequenceKeypoint.new(1, 1),
			}),
		}),
	})

	-- Main panel
	local main = make("Frame", {
		Name = "Main",
		Parent = window,
		Position = UDim2.fromOffset(184, 0),
		BackgroundTransparency = 1,
		Size = UDim2.new(1, -184, 1, 0),
	})
	local topBar = make("Frame", {
		Name = "TopBar",
		Parent = main,
		BackgroundTransparency = 1,
		Size = UDim2.new(1, 0, 0, 56),
	})
	refs.topBar = topBar
	refs.pageTitle = make("TextLabel", {
		Name = "Title",
		Parent = topBar,
		FontFace = font("Bold"),
		TextColor3 = theme.text,
		Text = "",
		BackgroundTransparency = 1,
		TextXAlignment = Enum.TextXAlignment.Left,
		Position = UDim2.fromOffset(52, 0),
		TextSize = 20,
		Size = UDim2.new(0, 240, 1, 0),
	})

	local searchBar = make("Frame", {
		Name = "SearchBar",
		Parent = topBar,
		AnchorPoint = Vector2.new(1, 0.5),
		Position = UDim2.new(1, -24, 0.5, 0),
		BackgroundColor3 = theme.card,
		Size = UDim2.fromOffset(220, 32),
		Attr = { PlopacleNode = "search-bar" },
	}, { corner(0, 6) })
	refs.searchBar = searchBar
	refs.searchStroke = make("UIStroke", {
		Name = "Stroke",
		Parent = searchBar,
		Color = theme.border,
		Transparency = 0.5,
		Attr = { PlopacleNode = "search-stroke" },
	})
	refs.searchInput = make("TextBox", {
		Name = "Input",
		Parent = searchBar,
		FontFace = font("Regular"),
		ClipsDescendants = true,
		TextColor3 = theme.text,
		Text = "",
		ClearTextOnFocus = false,
		Position = UDim2.fromOffset(34, 0),
		TextXAlignment = Enum.TextXAlignment.Left,
		BackgroundTransparency = 1,
		PlaceholderColor3 = theme.textDim,
		TextSize = 14,
		PlaceholderText = "Search everything",
		Size = UDim2.new(1, -46, 1, 0),
	})
	local searchIcon = make("CanvasGroup", {
		Name = "Icon",
		Parent = searchBar,
		AnchorPoint = Vector2.new(0, 0.5),
		BackgroundTransparency = 1,
		Position = UDim2.new(0, 10, 0.5, 0),
		GroupColor3 = theme.textDim,
		Size = UDim2.fromOffset(16, 16),
	})
	refs.searchIcon = searchIcon
	make("Frame", {
		Name = "Ring",
		Parent = searchIcon,
		Position = UDim2.fromOffset(2, 2),
		BackgroundTransparency = 1,
		Size = UDim2.fromOffset(8, 8),
	}, {
		fullCorner(),
		make("UIStroke", { Color = theme.white, Thickness = 1.5, ApplyStrokeMode = Enum.ApplyStrokeMode.Border }),
	})
	make("Frame", {
		Name = "Handle",
		Parent = searchIcon,
		Rotation = 45,
		Position = UDim2.fromOffset(13, 13),
		BackgroundColor3 = theme.white,
		AnchorPoint = Vector2.new(0.5, 0.5),
		Size = UDim2.fromOffset(5, 2),
	}, { fullCorner() })

	refs.titleIcon = make("Frame", {
		Name = "TitleIcon",
		Parent = topBar,
		AnchorPoint = Vector2.new(0, 0.5),
		BackgroundTransparency = 1,
		Position = UDim2.new(0, 24, 0.5, 0),
		Size = UDim2.fromOffset(20, 20),
	})

	make("Frame", {
		Name = "Divider",
		Parent = main,
		BackgroundTransparency = 0.5,
		Position = UDim2.fromOffset(0, 56),
		BackgroundColor3 = theme.border,
		Size = UDim2.new(1, 0, 0, 1),
	})
	local pageHost = make("CanvasGroup", {
		Name = "PageHost",
		Parent = main,
		Position = UDim2.fromOffset(0, 57),
		BackgroundTransparency = 1,
		Size = UDim2.new(1, 0, 1, -57),
		Attr = { PlopacleNode = "page-host" },
	})
	refs.pageHost = pageHost

	-- Search results
	local results = scroller(theme, "SearchResults", nil, 8)
	results.Parent = pageHost
	refs.results = results
	refs.resultsEmpty = make("TextLabel", {
		Name = "Empty",
		Parent = results,
		Visible = false,
		FontFace = font("Regular"),
		TextColor3 = theme.textDim,
		Text = "Nothing matches that search",
		BackgroundTransparency = 1,
		TextSize = 14,
		Size = UDim2.new(1, 0, 0, 48),
	})
	local resultTemplate = card(theme, "ResultTemplate", 1, 10)
	resultTemplate.Visible = false
	resultTemplate.Parent = results
	refs.resultTemplate = resultTemplate
	local head = make("Frame", {
		Name = "Head",
		Parent = resultTemplate,
		LayoutOrder = -1,
		BackgroundTransparency = 1,
		Size = UDim2.new(1, 0, 0, 20),
	})
	make("TextLabel", {
		Name = "Path",
		Parent = head,
		FontFace = font("Bold"),
		TextColor3 = theme.textDim,
		Text = "Page  /  Section",
		BackgroundTransparency = 1,
		TextXAlignment = Enum.TextXAlignment.Left,
		TextSize = 13,
		Size = UDim2.new(1, -96, 1, 0),
	})
	make("TextButton", {
		Name = "Jump",
		Parent = head,
		FontFace = font("Medium"),
		TextColor3 = theme.textDim,
		Text = "Go to page",
		AutoButtonColor = false,
		AnchorPoint = Vector2.new(1, 0.5),
		BackgroundTransparency = 1,
		TextXAlignment = Enum.TextXAlignment.Right,
		Position = UDim2.new(1, 0, 0.5, 0),
		TextSize = 13,
		Size = UDim2.new(0, 88, 1, 0),
		Attr = { PlopacleNode = "result-jump" },
	})

	edge(theme, 10, 0.4, 9).Parent = window

	-- Overlay layer: dropdown menus render here so the page's scrolling frame cannot clip them.
	local overlay = make("Frame", {
		Name = "Overlay",
		Parent = window,
		Visible = false,
		BackgroundTransparency = 1,
		ZIndex = 20,
		Size = UDim2.fromScale(1, 1),
	})
	refs.overlay = overlay
	refs.catcher = make("TextButton", {
		Name = "Catcher",
		Parent = overlay,
		BackgroundTransparency = 1,
		AutoButtonColor = false,
		Text = "",
		Size = UDim2.fromScale(1, 1),
	})
	local dropdownMenu = make("CanvasGroup", {
		Name = "DropdownMenu",
		Parent = overlay,
		BackgroundColor3 = theme.menu,
		ZIndex = 2,
		AutomaticSize = Enum.AutomaticSize.Y,
		Size = UDim2.fromOffset(160, 0),
		Attr = { PlopacleNode = "dropdown-menu" },
	}, { corner(0, 8) })
	refs.dropdownMenu = dropdownMenu
	-- Body: the menu's content at full height. The menu is a window onto it that unrolls.
	local menuBody = make("Frame", {
		Name = "Body",
		Parent = dropdownMenu,
		BackgroundTransparency = 1,
		Size = UDim2.new(1, 0, 0, 0),
	})
	refs.menuBody = menuBody
	local menuSearch = make("Frame", {
		Name = "Search",
		Parent = menuBody,
		BackgroundColor3 = theme.bg,
		Position = UDim2.fromOffset(4, 4),
		Size = UDim2.new(1, -8, 0, MENU_SEARCH_HEIGHT - 4),
	}, { corner(0, 6) })
	refs.menuSearch = menuSearch
	refs.menuSearchStroke = make("UIStroke", {
		Name = "Stroke",
		Parent = menuSearch,
		Color = theme.border,
		Transparency = 0.4,
		Attr = { PlopacleNode = "field-stroke" },
	})
	refs.menuSearchInput = make("TextBox", {
		Name = "Input",
		Parent = menuSearch,
		FontFace = font("Regular"),
		ClipsDescendants = true,
		TextColor3 = theme.text,
		Text = "",
		ClearTextOnFocus = false,
		Position = UDim2.fromOffset(8, 0),
		TextXAlignment = Enum.TextXAlignment.Left,
		BackgroundTransparency = 1,
		PlaceholderColor3 = theme.textDim,
		TextSize = 13,
		PlaceholderText = "Search",
		Size = UDim2.new(1, -16, 1, 0),
	})
	local menuList = make("ScrollingFrame", {
		Name = "List",
		Parent = menuBody,
		ScrollBarThickness = 3,
		BackgroundTransparency = 1,
		ScrollingDirection = Enum.ScrollingDirection.Y,
		CanvasSize = UDim2.new(0, 0, 0, 0),
		ScrollBarImageColor3 = theme.border,
		Size = UDim2.fromScale(1, 1),
	}, {
		make("UIPadding", {
			PaddingBottom = UDim.new(0, 4),
			PaddingTop = UDim.new(0, 4),
			PaddingLeft = UDim.new(0, 4),
			PaddingRight = UDim.new(0, 4),
		}),
		make("UIListLayout", { Padding = UDim.new(0, 2), SortOrder = Enum.SortOrder.LayoutOrder }),
	})
	refs.menuList = menuList
	refs.menuEmpty = make("TextLabel", {
		Name = "Empty",
		Parent = menuBody,
		Visible = false,
		FontFace = font("Regular"),
		TextColor3 = theme.textDim,
		Text = "No matches",
		BackgroundTransparency = 1,
		TextSize = 13,
		Size = UDim2.new(1, 0, 0, OPTION_HEIGHT + 8),
	})
	local check = ICONS.Check
	refs.optionTemplate = make("TextButton", {
		Name = "OptionTemplate",
		Parent = menuList,
		Visible = false,
		FontFace = font("Medium"),
		TextColor3 = theme.text,
		Text = "Option",
		AutoButtonColor = false,
		BackgroundColor3 = theme.optionHover,
		BackgroundTransparency = 1,
		TextXAlignment = Enum.TextXAlignment.Left,
		TextTruncate = Enum.TextTruncate.AtEnd,
		TextSize = 14,
		Size = UDim2.new(1, 0, 0, OPTION_HEIGHT),
		Attr = { PlopacleNode = "option" },
	}, {
		corner(0, 6),
		make("UIPadding", { PaddingLeft = UDim.new(0, 10), PaddingRight = UDim.new(0, 10) }),
		make("ImageLabel", {
			Name = "Check",
			Visible = false,
			ScaleType = Enum.ScaleType.Fit,
			ImageColor3 = theme.accent,
			AnchorPoint = Vector2.new(1, 0.5),
			ImageRectOffset = Vector2.new(check[2], check[3]),
			BackgroundTransparency = 1,
			ImageRectSize = Vector2.new(check[4], check[4]),
			Position = UDim2.new(1, 0, 0.5, 0),
			Image = "rbxassetid://" .. check[1],
			Size = UDim2.fromOffset(16, 16),
		}),
	})
	edge(theme, 3, 0.2, 7).Parent = dropdownMenu

	-- Notifications sit outside the window so they show while it is hidden.
	local toasts = make("Frame", {
		Name = "Toasts",
		Parent = gui,
		AnchorPoint = Vector2.new(1, 1),
		BackgroundTransparency = 1,
		Position = UDim2.new(1, -24, 1, -24),
		Size = UDim2.new(0, 320, 1, -48),
	}, {
		make("UIListLayout", {
			VerticalAlignment = Enum.VerticalAlignment.Bottom,
			SortOrder = Enum.SortOrder.LayoutOrder,
			HorizontalAlignment = Enum.HorizontalAlignment.Right,
			Padding = UDim.new(0, 8),
		}),
	})
	refs.toasts = toasts
	local toastTemplate = make("CanvasGroup", {
		Name = "ToastTemplate",
		Parent = toasts,
		Visible = false,
		BackgroundColor3 = theme.card,
		AutomaticSize = Enum.AutomaticSize.Y,
		Size = UDim2.new(1, 0, 0, 0),
		Attr = { PlopacleNode = "toast" },
	}, {
		corner(0, 8),
		make("UIScale", { Name = "Scale", Attr = { PlopacleNode = "toast-scale" } }),
		edge(theme, 3, 0.3, 7),
	})
	refs.toastTemplate = toastTemplate
	local info = ICONS.Info
	make("ImageLabel", {
		Name = "Icon",
		Parent = toastTemplate,
		ImageColor3 = theme.accent,
		ScaleType = Enum.ScaleType.Fit,
		ImageRectOffset = Vector2.new(info[2], info[3]),
		BackgroundTransparency = 1,
		Position = UDim2.fromOffset(14, 13),
		ImageRectSize = Vector2.new(info[4], info[4]),
		Image = "rbxassetid://" .. info[1],
		Size = UDim2.fromOffset(20, 20),
	})
	make("Frame", {
		Name = "Body",
		Parent = toastTemplate,
		AutomaticSize = Enum.AutomaticSize.Y,
		BackgroundTransparency = 1,
		Size = UDim2.new(1, 0, 0, 0),
	}, {
		make("UIPadding", {
			PaddingBottom = UDim.new(0, 12),
			PaddingTop = UDim.new(0, 12),
			PaddingLeft = UDim.new(0, 46),
			PaddingRight = UDim.new(0, 14),
		}),
		make("UIListLayout", { Padding = UDim.new(0, 2), SortOrder = Enum.SortOrder.LayoutOrder }),
		make("TextLabel", {
			Name = "Title",
			LayoutOrder = 1,
			FontFace = font("Bold"),
			TextColor3 = theme.text,
			Text = "Notification",
			BackgroundTransparency = 1,
			TextXAlignment = Enum.TextXAlignment.Left,
			TextTruncate = Enum.TextTruncate.AtEnd,
			TextSize = 15,
			Size = UDim2.new(1, 0, 0, 20),
		}),
		make("TextLabel", {
			Name = "Message",
			LayoutOrder = 2,
			TextWrapped = true,
			TextColor3 = theme.textDim,
			Text = "Message text",
			TextSize = 13,
			BackgroundTransparency = 1,
			TextXAlignment = Enum.TextXAlignment.Left,
			FontFace = font("Regular"),
			TextYAlignment = Enum.TextYAlignment.Top,
			AutomaticSize = Enum.AutomaticSize.Y,
			Size = UDim2.new(1, 0, 0, 0),
		}),
	})

	return refs
end

---------------------------------------------------------------------------------------------
-- Window

local function resolveParent(options)
	if options.parent then
		return options.parent
	end
	local gethui = executorGlobal("gethui")
	if type(gethui) == "function" then
		local ok, hidden = pcall(gethui)
		if ok and hidden then
			return hidden
		end
	end
	local ok, coreGui = pcall(function()
		local service = game:GetService("CoreGui")
		local _ = service.Name
		return service
	end)
	if ok and coreGui and RunService:IsStudio() == false then
		return coreGui
	end
	return Players.LocalPlayer:WaitForChild("PlayerGui")
end

--[=[
	Plorch.new(options) -> window

	options: title, subtitle, logo, name, size (Vector2), toggleKey (Enum.KeyCode),
	theme ({ token = Color3 }), parent, displayOrder, settingsPage (default true),
	configFolder (default "Plorch"), autoSave (default false),
	autoScale (default true: the window is sized for a 1080-line screen and grows or shrinks
	with the viewport), scale (default 1: a factor on the window's size, from 0.5 to 2, on top
	of autoScale), mobileButton (default: on touch devices with no keyboard; a small
	draggable button that opens and closes the window)
]=]
function Plorch.new(options)
	options = options or {}
	local static = options._static == true -- build only: no input, no animation (verify_export)

	local theme = {}
	for token, hex in pairs(THEME) do
		theme[token] = (options.theme and options.theme[token]) or Color3.fromHex(hex)
	end

	local motion = newMotion(theme)
	local entry, remember, toGoal, toRest, playFrom, bindHover =
		motion.entry, motion.remember, motion.toGoal, motion.toRest, motion.playFrom, motion.bindHover

	local refs = buildShell(theme, options)
	local gui, frame, windowScale = refs.gui, refs.window, refs.windowScale
	local pageHost, results, overlay = refs.pageHost, refs.results, refs.overlay
	local dropdownMenu, menuList = refs.dropdownMenu, refs.menuList
	local searchInput = refs.searchInput

	local window = { instance = gui, refs = refs, theme = theme, flags = {} }
	local connections = {} -- global connections, dropped on destroy
	local unloadCallbacks = {}
	local destroyed = false

	local function track(connection)
		table.insert(connections, connection)
		return connection
	end

	-----------------------------------------------------------------------------------------
	-- Notifications

	local toastOrder = 0

	-- window:notify("Title", "Message", seconds) or window:notify({ title, message, duration })
	function window:notify(title, message, duration)
		if type(title) == "table" then
			title, message, duration = title.title, title.message, title.duration
		end
		if destroyed then
			return
		end
		ui(function()
			toastOrder += 1
			local toast = refs.toastTemplate:Clone()
			local scale = toast:FindFirstChild("Scale")
			local body = toast:FindFirstChild("Body")
			toast.Name = "Toast"
			toast.LayoutOrder = toastOrder
			body.Title.Text = tostring(title or "")
			body.Message.Text = message and tostring(message) or ""
			body.Message.Visible = message ~= nil and message ~= ""
			toast.Visible = true
			toast.Parent = refs.toasts
			playFrom(toast, "toast", "enter")
			playFrom(scale, "toast-scale", "enter")
			task.delay(duration or TOAST_DURATION, function()
				toGoal(toast, "toast", "exit")
				toGoal(scale, "toast-scale", "exit")
				task.wait(TOAST_EXIT_TIME)
				spr.stop(toast)
				spr.stop(scale)
				toast:Destroy()
			end)
		end)
	end

	-----------------------------------------------------------------------------------------
	-- Open / close

	local isOpen = false
	local openToken = 0
	local toggleKey = parseKey(options.toggleKey) or Enum.KeyCode.RightControl
	local listeningBind -- the keybind row waiting for a key press, if any
	local keybinds = {} -- every keybind row, for onPressed dispatch
	local closeDropdown -- assigned in the dropdown section

	-- A centred window lands on half pixels when the viewport is odd-sized, which blurs
	-- everything drawn through the CanvasGroup. Nudge it onto whole pixels (at rest only:
	-- the open/close scale moves AbsolutePosition).
	-- The window is drawn for a 1080-line screen. `fit` is what its UIScale rests at on this
	-- one: in proportion to the viewport's height, as the game's own interface is, and never
	-- so large that the window leaves the viewport. A phone is the exception: in proportion
	-- the window would be too small to touch, so there it is as large as fits.
	local fit = 1
	local ownScale = math.clamp(tonumber(options.scale) or 1, 0.5, 2)
	local function measureFit()
		local camera = workspace.CurrentCamera
		local viewport = camera and camera.ViewportSize
		if static then
			return 1
		end
		if options.autoScale == false or not viewport or viewport.X < 1 or viewport.Y < 1 then
			return ownScale
		end
		local size = frame.Size
		local wanted = viewport.Y / REFERENCE_HEIGHT * ownScale
		if UserInputService.TouchEnabled and not UserInputService.KeyboardEnabled then
			wanted = math.max(wanted, 1)
		end
		wanted = math.min(wanted, MAX_FIT)
		local room = math.min((viewport.X - SCREEN_MARGIN) / size.X.Offset, (viewport.Y - SCREEN_MARGIN) / size.Y.Offset)
		-- In steps of a twentieth, so a window sized in round numbers lands on whole pixels
		-- and stays sharp through the CanvasGroup.
		return math.max(math.floor(math.min(wanted, room) * 20 + 0.001) / 20, MIN_FIT)
	end
	fit = measureFit()
	windowScale.Scale = fit

	-- Screen pixels to the window's own units, for anything measured with Absolute*.
	local function unscaled()
		return windowScale.Scale > 0 and windowScale.Scale or 1
	end

	-- The window's place is kept as a fraction of the screen, so it moves with the viewport
	-- when that is resized; the offset only ever holds the whole-pixel nudge below.
	local function positionAsFraction()
		local screen = gui.AbsoluteSize
		if screen.X < 1 or screen.Y < 1 then
			return
		end
		local position = frame.Position
		frame.Position = UDim2.fromScale(
			position.X.Scale + position.X.Offset / screen.X,
			position.Y.Scale + position.Y.Offset / screen.Y
		)
	end

	local function snapToPixels()
		if math.abs(windowScale.Scale - fit) > 0.001 then
			return
		end
		local position = frame.AbsolutePosition
		local dx, dy = math.round(position.X) - position.X, math.round(position.Y) - position.Y
		if dx ~= 0 or dy ~= 0 then
			frame.Position += UDim2.fromOffset(dx, dy)
		end
	end

	function window:setOpen(open)
		open = open == true
		if open == isOpen or destroyed then
			return
		end
		isOpen = open
		openToken += 1
		local token = openToken
		if open then
			local wasHidden = not frame.Visible
			frame.Visible = true
			playFrom(frame, "window", "open", not wasHidden)
			-- The scale springs by hand: its rest is `fit`, not the 1 the tuning assumes.
			local opening = entry("window-scale", "open")
			if wasHidden then
				spr.stop(windowScale)
				windowScale.Scale = opening.from.Scale * fit
			end
			spr.target(windowScale, opening.damping, opening.frequency, { Scale = fit })
		else
			searchInput:ReleaseFocus()
			closeDropdown()
			if listeningBind then
				listeningBind.cancel()
			end
			toGoal(frame, "window", "close")
			local closing = entry("window-scale", "close")
			spr.target(windowScale, closing.damping, closing.frequency, { Scale = closing.goal.Scale * fit })
			task.delay(CLOSE_HIDE_DELAY, function()
				if openToken == token then
					frame.Visible = false
				end
			end)
		end
	end

	function window:isOpen()
		return isOpen
	end

	function window:toggle()
		self:setOpen(not isOpen)
	end

	local function setToggleKey(code)
		toggleKey = code
	end

	-----------------------------------------------------------------------------------------
	-- Tabs and pages

	local pages = {} -- name -> page object
	local pageOrder = 0
	local currentPage
	local searching = false

	local function applyTabState(page, active, instant)
		local state = active and "active" or "inactive"
		local tab = page.tab
		toGoal(tab.button, "tab", state, instant)
		toGoal(tab.indicator, "tab-indicator", state, instant)
		toGoal(tab.label, "tab-label", state, instant)
		toGoal(tab.icon, "tab-icon", state, instant)
	end

	-- The top bar shows a copy of the current tab's icon beside the page title.
	local function setTitleIcon(source)
		if static then
			return
		end
		refs.titleIcon:ClearAllChildren()
		if not source then
			return
		end
		local icon = source:Clone()
		icon:SetAttribute("PlopacleNode", nil)
		icon.AnchorPoint = Vector2.new(0.5, 0.5)
		icon.Position = UDim2.fromScale(0.5, 0.5)
		icon.GroupColor3 = theme.accent
		icon.Parent = refs.titleIcon
	end

	local function showPage(name)
		closeDropdown()
		for pageName, page in pairs(pages) do
			page.instance.Visible = pageName == name
		end
		results.Visible = false
		refs.pageTitle.Text = name
		setTitleIcon(pages[name] and pages[name].tab.icon)
		if not static then
			playFrom(pageHost, "page-host", "enter")
		end
	end

	local function selectPage(name, instant)
		local target = pages[name]
		if not target then
			return
		end
		if currentPage == name and not searching then
			return
		end
		currentPage = name
		for pageName, page in pairs(pages) do
			applyTabState(page, pageName == name, instant)
		end
		target.instance.CanvasPosition = Vector2.zero
		showPage(name)
	end

	function window:selectPage(name)
		if searching then
			searchInput.Text = ""
		end
		selectPage(name)
	end

	-----------------------------------------------------------------------------------------
	-- Dropdown menu (one shared menu in the window's overlay layer)

	local openMenu -- the dropdown whose menu is showing
	local menuToken = 0
	local menuFollow -- per-frame connection that keeps the open menu attached to its field

	local function placeMenu(dropdown, opensUp)
		local button = dropdown.button
		local scale = unscaled()
		local origin = (button.AbsolutePosition - overlay.AbsolutePosition) / scale
		local y = opensUp and origin.Y - 4 or origin.Y + button.AbsoluteSize.Y / scale + 4
		dropdownMenu.Position = UDim2.fromOffset(math.round(origin.X), math.round(y))
	end

	-- False once the field has been scrolled out of the page area.
	local function fieldInView(dropdown)
		local button = dropdown.button
		local middle = button.AbsolutePosition.Y + button.AbsoluteSize.Y / 2
		local top = pageHost.AbsolutePosition.Y
		return middle >= top and middle <= top + pageHost.AbsoluteSize.Y
	end

	local menuBody, menuSearch, menuSearchInput = refs.menuBody, refs.menuSearch, refs.menuSearchInput
	local menuItems = {} -- { button, option, text, rest, from } for the open menu, in option order
	local menuWidth, menuSearchHeight = 160, 0
	local settingSearch = false -- true while the search text is being reset by the library

	local function clearOptions()
		for _, item in ipairs(menuItems) do
			item.button:Destroy()
		end
		table.clear(menuItems)
	end

	local function listHeight(count)
		return count * OPTION_HEIGHT + math.max(count - 1, 0) * 2 + 8
	end

	-- An option fades in from the `option.enter` start values.
	local function enterOption(item, delay)
		local enter = entry("option", "enter")
		spr.stop(item.button)
		for prop, value in pairs(item.from) do
			item.button[prop] = value
		end
		task.delay(delay, function()
			if item.button.Parent then
				spr.target(item.button, enter.damping, enter.frequency, item.rest)
			end
		end)
	end

	-- Shows the options that match `query`, sizes the list to at most MENU_MAX_VISIBLE of
	-- them (the rest scroll) and returns the menu's full height.
	local function layoutMenu(query, animate)
		local matches = 0
		for _, item in ipairs(menuItems) do
			local match = query == "" or item.text:find(query, 1, true) ~= nil
			if match then
				matches += 1
				if animate and not item.button.Visible then
					enterOption(item, math.min(matches - 1, MENU_MAX_VISIBLE) * OPTION_STAGGER)
				end
			end
			item.button.Visible = match
		end
		refs.menuEmpty.Visible = matches == 0
		menuList.CanvasSize = UDim2.fromOffset(0, matches == 0 and 0 or listHeight(matches))
		local height = matches == 0 and OPTION_HEIGHT + 8 or listHeight(math.min(matches, MENU_MAX_VISIBLE))
		return menuSearchHeight + height
	end

	-- The menu rolls back up into its field and fades; clicks pass through straight away.
	closeDropdown = function()
		local dropdown = openMenu
		if not dropdown then
			return
		end
		openMenu = nil
		menuToken += 1
		local token = menuToken
		if menuFollow then
			menuFollow:Disconnect()
			menuFollow = nil
		end
		menuSearchInput:ReleaseFocus()
		refs.catcher.Visible = false
		toRest(dropdown.chevron, "dropdown-chevron", "open")
		toRest(dropdown.stroke, "dropdown-stroke", "open")
		toGoal(dropdownMenu, "dropdown-menu", "close")
		local shrink = entry("dropdown-menu", "shrink")
		spr.target(dropdownMenu, shrink.damping, shrink.frequency, {
			Size = UDim2.fromOffset(dropdownMenu.Size.X.Offset, MENU_COLLAPSED_HEIGHT),
		})
		task.delay(MENU_CLOSE_TIME, function()
			if menuToken == token then
				overlay.Visible = false
				clearOptions()
			end
		end)
	end

	local function openDropdown(dropdown)
		closeDropdown()
		menuToken += 1
		clearOptions()
		openMenu = dropdown

		-- The search field sits above the list unless the dropdown opted out.
		local hasSearch = dropdown.search ~= false
		menuSearchHeight = hasSearch and MENU_SEARCH_HEIGHT or 0
		menuSearch.Visible = hasSearch
		settingSearch = true
		menuSearchInput.Text = ""
		settingSearch = false
		menuList.Position = UDim2.fromOffset(0, menuSearchHeight)
		menuList.Size = UDim2.new(1, 0, 1, -menuSearchHeight)
		refs.menuEmpty.Position = UDim2.fromOffset(0, menuSearchHeight)

		local multi = dropdown.toggle ~= nil
		local current = dropdown.get()
		local count = #dropdown.options
		-- Open scrolled so the current choice is in view, a couple of options down.
		local currentIndex = not multi and table.find(dropdown.options, current) or 1
		local skipped = math.clamp(currentIndex - 3, 0, math.max(count - MENU_MAX_VISIBLE, 0))
		local from = motion.decodeProps(entry("option", "enter").from)
		for index, option in ipairs(dropdown.options) do
			local button = refs.optionTemplate:Clone()
			button.Name = "Option_" .. index
			button:SetAttribute("Option", true)
			button.LayoutOrder = index
			button.Text = tostring(option)
			-- A multi-select option shows a tick when on and stays open when clicked.
			if multi then
				button.Check.Visible = dropdown.isOn(option)
			else
				button.Check.Visible = option == current
				if option == current then
					button.BackgroundTransparency = OPTION_SELECTED_TRANSPARENCY
				end
			end
			button.Visible = true
			button.Parent = menuList
			bindHover(button, "option")
			button.MouseButton1Click:Connect(function()
				if multi then
					dropdown.toggle(option)
					button.Check.Visible = dropdown.isOn(option)
					return
				end
				dropdown.set(option)
				closeDropdown()
			end)
			local rest = {}
			for prop in pairs(from) do
				rest[prop] = button[prop]
			end
			local item = { button = button, option = option, text = tostring(option):lower(), rest = rest, from = from }
			table.insert(menuItems, item)
			-- Options fade in one after another as the menu unrolls past them.
			enterOption(item, math.clamp(index - skipped - 1, 0, MENU_MAX_VISIBLE) * OPTION_STAGGER)
		end

		-- Unroll downward from the field, or upward when the window's bottom edge is in the way.
		local height = layoutMenu("", false)
		local button = dropdown.button
		local scale = unscaled()
		local origin = (button.AbsolutePosition - overlay.AbsolutePosition) / scale
		local width = button.AbsoluteSize.X / scale
		menuWidth = width
		local below = origin.Y + button.AbsoluteSize.Y / scale + 4
		local opensUp = below + height > overlay.AbsoluteSize.Y / scale - 8
		local anchor = opensUp and 1 or 0
		spr.stop(dropdownMenu)
		dropdownMenu.AutomaticSize = Enum.AutomaticSize.None
		dropdownMenu.AnchorPoint = Vector2.new(0, anchor)
		placeMenu(dropdown, opensUp)
		dropdownMenu.Size = UDim2.fromOffset(width, MENU_COLLAPSED_HEIGHT)
		menuBody.AnchorPoint = Vector2.new(0, anchor)
		menuBody.Position = UDim2.fromScale(0, anchor)
		menuBody.Size = UDim2.new(1, 0, 0, height)
		refs.catcher.Visible = true
		overlay.Visible = true
		-- A canvas position is in screen pixels, unlike the sizes around it.
		local scroll = Vector2.new(0, skipped * (OPTION_HEIGHT + 2) * scale)
		menuList.CanvasPosition = scroll
		local token = menuToken
		task.defer(function()
			-- Again once the list has its size; a position past the old window is clamped away.
			if menuToken == token then
				menuList.CanvasPosition = scroll
			end
		end)
		playFrom(dropdownMenu, "dropdown-menu", "open")
		local grow = entry("dropdown-menu", "grow")
		spr.target(dropdownMenu, grow.damping, grow.frequency, { Size = UDim2.fromOffset(width, height) })
		toGoal(dropdown.chevron, "dropdown-chevron", "open")
		toGoal(dropdown.stroke, "dropdown-stroke", "open")

		-- Scrolling the page moves the field; the menu rides along, and closes if the field
		-- leaves the page area.
		menuFollow = RunService.RenderStepped:Connect(function()
			if fieldInView(dropdown) then
				placeMenu(dropdown, opensUp)
			else
				closeDropdown()
			end
		end)
	end

	refs.catcher.MouseButton1Click:Connect(function()
		closeDropdown()
	end)

	-- Typing filters the open menu; the menu springs to the height of what is left.
	menuSearchInput:GetPropertyChangedSignal("Text"):Connect(function()
		if settingSearch or not openMenu then
			return
		end
		local query = menuSearchInput.Text:lower():match("^%s*(.-)%s*$")
		local height = layoutMenu(query, true)
		menuList.CanvasPosition = Vector2.zero
		menuBody.Size = UDim2.new(1, 0, 0, height)
		local grow = entry("dropdown-menu", "grow")
		spr.target(dropdownMenu, grow.damping, grow.frequency, { Size = UDim2.fromOffset(menuWidth, height) })
	end)
	remember(refs.menuSearchStroke, "field-stroke", "focus")
	menuSearchInput.Focused:Connect(function()
		toGoal(refs.menuSearchStroke, "field-stroke", "focus")
	end)
	-- Enter picks the first match.
	menuSearchInput.FocusLost:Connect(function(enterPressed)
		toRest(refs.menuSearchStroke, "field-stroke", "focus")
		local dropdown = openMenu
		if not (enterPressed and dropdown) then
			return
		end
		for _, item in ipairs(menuItems) do
			if item.button.Visible then
				if dropdown.toggle then
					dropdown.toggle(item.option)
					item.button.Check.Visible = dropdown.isOn(item.option)
					return
				end
				dropdown.set(item.option)
				closeDropdown()
				return
			end
		end
	end)

	-----------------------------------------------------------------------------------------
	-- Config (flags)

	local saveQueued = false
	local autoSave = options.autoSave == true
	local configFolder = options.configFolder or "Plorch"
	local loadingConfig = false

	local activeConfig = "default" -- the config autosave writes to
	local configChanged -- assigned by the Settings page: refreshes its config picker

	local function configPath(name)
		return configFolder .. "/" .. name .. ".json"
	end

	-- A config name as it may be used for a file: letters, digits, spaces, - and _.
	local function cleanName(name)
		local cleaned = tostring(name or ""):gsub("[^%w _%-]", "")
		return cleaned:match("^%s*(.-)%s*$"):sub(1, 32)
	end

	-- Makes the config folder one level at a time (makefolder does not nest everywhere).
	local function ensureFolder(files)
		if type(files.makefolder) ~= "function" or type(files.isfolder) ~= "function" then
			return
		end
		local path = ""
		for part in configFolder:gmatch("[^/]+") do
			path = path == "" and part or (path .. "/" .. part)
			if not files.isfolder(path) then
				files.makefolder(path)
			end
		end
	end

	local function configExists(files, name)
		local ok, found = pcall(files.isfile, configPath(name))
		return ok and found == true
	end

	-- The saved form of a config. Sharing will send this same payload, so it carries its own
	-- format version and name instead of relying on the file it sits in.
	local function encodeConfig(name)
		local values = {}
		for flag, control in pairs(window.flags) do
			local value = control.get()
			if typeof(value) == "EnumItem" then
				value = value.Name
			end
			values[flag] = value
		end
		return HttpService:JSONEncode({ plorch = CONFIG_FORMAT, name = name, values = values })
	end

	-- Payload text -> { name, values }, or nil. Files written before the format carried a
	-- version are a bare flag table.
	local function decodeConfig(text)
		local ok, data = pcall(function()
			return HttpService:JSONDecode(text)
		end)
		if not ok or type(data) ~= "table" then
			return nil
		end
		if type(data.plorch) == "number" and type(data.values) == "table" then
			return { name = data.name, values = data.values }
		end
		return { values = data }
	end

	local function applyConfig(values)
		loadingConfig = true
		for flag, value in pairs(values) do
			local control = window.flags[flag]
			if control then
				pcall(control.set, value)
			end
		end
		loadingConfig = false
	end

	-- Makes `name` the config in use and remembers it for the next session.
	local function setActive(files, name)
		activeConfig = name
		pcall(function()
			ensureFolder(files)
			files.writefile(configFolder .. "/active.txt", name)
		end)
		if configChanged then
			configChanged()
		end
	end

	-- window:saveConfig(name?) -> boolean. Writes every flagged control's value to a file;
	-- with no name, to the config in use.
	function window:saveConfig(name)
		local files = fileApi()
		if not files then
			return false
		end
		name = name ~= nil and cleanName(name) or activeConfig
		if name == "" then
			return false
		end
		return (pcall(function()
			ensureFolder(files)
			files.writefile(configPath(name), encodeConfig(name))
		end))
	end

	-- window:loadConfig(name?) -> boolean. Applies saved values (each fires its callback) and
	-- makes that config the one in use. With no name: the config used last session.
	function window:loadConfig(name)
		local files = fileApi()
		if not files then
			return false
		end
		if name == nil then
			local ok, last = pcall(function()
				local path = configFolder .. "/active.txt"
				return files.isfile(path) and files.readfile(path) or nil
			end)
			name = ok and last and cleanName(last) or ""
			if name == "" or not configExists(files, name) then
				name = "default"
			end
		else
			name = cleanName(name)
		end
		local ok, config = pcall(function()
			if not files.isfile(configPath(name)) then
				return nil
			end
			return decodeConfig(files.readfile(configPath(name)))
		end)
		if not ok or not config then
			return false
		end
		setActive(files, name)
		applyConfig(config.values)
		return true
	end

	-- window:listConfigs() -> { name, ... }, sorted; always includes the config in use.
	function window:listConfigs()
		local files = fileApi()
		local names = {}
		if files and files.listfiles then
			pcall(function()
				for _, path in ipairs(files.listfiles(configFolder)) do
					local name = tostring(path):match("([^/\\]+)%.json$")
					if name and table.find(names, name) == nil then
						table.insert(names, name)
					end
				end
			end)
		end
		if table.find(names, activeConfig) == nil then
			table.insert(names, activeConfig)
		end
		table.sort(names)
		return names
	end

	function window:activeConfig()
		return activeConfig
	end

	-- window:createConfig(name) -> boolean, reason. Saves the current values under a new name
	-- and switches to it.
	function window:createConfig(name)
		local files = fileApi()
		if not files then
			return false, "This executor cannot write files"
		end
		name = cleanName(name)
		if name == "" then
			return false, "Type a name first"
		end
		if configExists(files, name) then
			return false, "A config with that name already exists"
		end
		if not window:saveConfig(name) then
			return false, "The file could not be written"
		end
		setActive(files, name)
		return true
	end

	-- window:renameConfig(name) -> boolean, reason. Renames the config in use.
	function window:renameConfig(name)
		local files = fileApi()
		if not files then
			return false, "This executor cannot write files"
		end
		if not files.delfile then
			return false, "This executor cannot delete files"
		end
		name = cleanName(name)
		local old = activeConfig
		if name == "" then
			return false, "Type a name first"
		end
		if name == old then
			return false, "That is already its name"
		end
		if configExists(files, name) then
			return false, "A config with that name already exists"
		end
		if not window:saveConfig(name) then
			return false, "The file could not be written"
		end
		pcall(files.delfile, configPath(old))
		setActive(files, name)
		return true
	end

	-- window:deleteConfig(name?) -> boolean, reason. Deletes a config; with no name, the one in
	-- use. Deleting the config in use switches to "default", which itself cannot be deleted.
	function window:deleteConfig(name)
		local files = fileApi()
		if not files then
			return false, "This executor cannot write files"
		end
		if not files.delfile then
			return false, "This executor cannot delete files"
		end
		name = name ~= nil and cleanName(name) or activeConfig
		if name == "default" then
			return false, "The default config cannot be deleted"
		end
		if name == "" or not configExists(files, name) then
			return false, "No config with that name"
		end
		if not pcall(files.delfile, configPath(name)) or configExists(files, name) then
			return false, "The file could not be deleted"
		end
		if name ~= activeConfig then
			if configChanged then
				configChanged()
			end
		elseif not window:loadConfig("default") then
			-- No default on disk yet: the current values become it.
			setActive(files, "default")
			window:saveConfig("default")
		end
		return true
	end

	function window:setAutoSave(enabled)
		autoSave = enabled == true
	end

	local function valueChanged()
		if not autoSave or loadingConfig or saveQueued then
			return
		end
		saveQueued = true
		task.delay(AUTOSAVE_DELAY, function()
			saveQueued = false
			if not destroyed then
				window:saveConfig()
			end
		end)
	end

	-----------------------------------------------------------------------------------------
	-- Search (covers every row on every page). A result card holds the real row, moved in
	-- from its section, so it works in place; leaving the search puts every row back home.

	local searchIndex = {} -- { row, home, title, description, page, section }
	local shown = {} -- { card, item } for the results on screen

	local function releaseResults()
		closeDropdown()
		for _, result in ipairs(shown) do
			result.item.row.Parent = result.item.home
			result.card:Destroy()
		end
		table.clear(shown)
	end

	local function jumpTo(item)
		searchInput:ReleaseFocus()
		searchInput.Text = ""
		selectPage(item.page)
		RunService.Heartbeat:Wait() -- let the page lay out before measuring
		local page = pages[item.page].instance
		local y = item.row.AbsolutePosition.Y - page.AbsolutePosition.Y + page.CanvasPosition.Y - 16 * unscaled()
		page.CanvasPosition = Vector2.new(0, math.max(0, y))
		local title = item.row:FindFirstChild("Title")
		if title then
			local flash = entry("row-title", "flash")
			spr.stop(title)
			title.TextColor3 = theme.accent2
			spr.target(title, flash.damping, flash.frequency, { TextColor3 = theme.text })
		end
	end

	local function runSearch(raw)
		local query = raw:lower():match("^%s*(.-)%s*$")
		if query == "" then
			if searching then
				searching = false
				releaseResults()
				if currentPage then
					showPage(currentPage)
				end
			end
			return
		end

		releaseResults()
		local count = 0
		for _, item in ipairs(searchIndex) do
			local haystack = (item.title .. " " .. item.description .. " " .. item.page .. " " .. item.section):lower()
			if item.row.Visible and not item.owner.hidden and haystack:find(query, 1, true) then
				count += 1
				local resultCard = refs.resultTemplate:Clone()
				local head = resultCard:FindFirstChild("Head")
				resultCard.Name = "Result_" .. item.row.Name
				resultCard.LayoutOrder = count
				head.Path.Text = item.page .. "  /  " .. item.section
				resultCard.Visible = true
				resultCard.Parent = results
				item.row.Parent = resultCard
				bindHover(head.Jump, "result-jump")
				head.Jump.MouseButton1Click:Connect(function()
					jumpTo(item)
				end)
				table.insert(shown, { card = resultCard, item = item })
				if count >= MAX_RESULTS then
					break
				end
			end
		end
		refs.resultsEmpty.Visible = count == 0
		results.CanvasPosition = Vector2.zero

		if not searching then
			searching = true
			for _, page in pairs(pages) do
				page.instance.Visible = false
			end
			results.Visible = true
			refs.pageTitle.Text = "Search"
			setTitleIcon(refs.searchIcon)
			playFrom(pageHost, "page-host", "enter")
		end
	end

	-----------------------------------------------------------------------------------------
	-- Rows

	local draggingSlider

	-- Shared surface of every row object: instance, setTitle, setDescription, setVisible.
	local function rowObject(row, item)
		local object = { instance = row }

		function object:setTitle(text)
			ui(function()
				row.Title.Text = tostring(text)
			end)
			if item then
				item.title = tostring(text)
			end
		end

		function object:setDescription(text)
			local description = row:FindFirstChild("Desc")
			if description then
				ui(function()
					description.Text = tostring(text)
				end)
			end
			if item then
				item.description = tostring(text)
			end
		end

		function object:setVisible(visible)
			ui(function()
				row.Visible = visible == true
			end)
		end

		return object
	end

	-- Value controls: get/set/onChanged, the `callback` option, flags and autosave.
	local function valueObject(object, config, flag, read, write)
		local listeners = {}

		function object:get()
			return read()
		end

		-- set(value, silent): silent skips the callback and listeners.
		function object:set(value, silent)
			local changed = write(value)
			if changed and not silent then
				local current = read()
				fire(config.callback, current)
				for _, listener in ipairs(listeners) do
					fire(listener, current)
				end
				valueChanged()
			end
		end

		function object:onChanged(listener)
			table.insert(listeners, listener)
		end

		if flag ~= false then
			window.flags[flag] = {
				get = read,
				set = function(value)
					object:set(value)
				end,
			}
		end
		return object
	end

	local function addSection(page, sectionConfig)
		if type(sectionConfig) == "string" then
			sectionConfig = { title = sectionConfig }
		end
		sectionConfig = sectionConfig or {}
		local sectionTitle = sectionConfig.title or "Section"
		local existing = page.sections[sectionTitle]
		if existing then
			return existing
		end

		page.sectionOrder += 1
		-- The holder is what the page lays out; it only takes a fixed height (and clips) while
		-- the section is sliding open or shut. Its 1 px padding keeps the card's outline, which
		-- is drawn outside the card, inside the clip, so the outline slides with the card. The
		-- gap to the next section is its bottom padding, not the page list's, so a section that
		-- collapses to nothing takes its gap with it and its neighbours never jump.
		local holder = make("Frame", {
			Name = "Section_" .. identifier(sectionTitle),
			Parent = page.instance,
			LayoutOrder = sectionConfig.order or page.sectionOrder,
			BackgroundTransparency = 1,
			AutomaticSize = Enum.AutomaticSize.Y,
			Size = UDim2.new(1, 0, 0, 0),
		}, {
			make("UIPadding", {
				PaddingBottom = UDim.new(0, 1 + SECTION_GAP),
				PaddingTop = UDim.new(0, 1),
				PaddingLeft = UDim.new(0, 1),
				PaddingRight = UDim.new(0, 1),
			}),
		})
		local sectionFrame = card(theme, "Card", 0, 12)
		sectionFrame.Parent = holder
		local header = make("TextLabel", {
			Name = "Header",
			Parent = sectionFrame,
			FontFace = font("Bold"),
			TextColor3 = theme.textDim,
			Text = sectionTitle,
			BackgroundTransparency = 1,
			TextXAlignment = Enum.TextXAlignment.Left,
			TextSize = 13,
			Size = UDim2.new(1, 0, 0, 20),
		})

		local section = { instance = holder, card = sectionFrame, page = page, title = sectionTitle, hidden = false }
		page.sections[sectionTitle] = section
		local rowOrder = 0

		-- Common bookkeeping for a new row: id, order, flag, search entry.
		local function prepare(config, kind)
			config = config or {}
			rowOrder += 1
			local id = config.id or identifier(config.title or (kind .. rowOrder))
			local flag = config.flag
			if flag == nil then
				flag = page.name .. "." .. id
			end
			return config, id, config.order or rowOrder, flag
		end

		local function register(row, config)
			row.Parent = sectionFrame
			local item = {
				row = row,
				home = sectionFrame,
				title = config.title or "",
				description = config.description or "",
				page = page.name,
				section = sectionTitle,
				owner = section,
			}
			table.insert(searchIndex, item)
			return item
		end

		function section:setTitle(text)
			ui(function()
				header.Text = tostring(text)
			end)
		end

		local visibilityToken = 0

		local function settle(visible)
			holder.AutomaticSize = Enum.AutomaticSize.Y
			holder.ClipsDescendants = false
			holder.Size = UDim2.new(1, 0, 0, 0)
			holder.Visible = visible
		end

		-- section:setVisible(bool, instant?): slides the section open or shut. A hidden
		-- section's rows are left out of search.
		function section:setVisible(visible, instant)
			visible = visible == true
			if section.hidden == not visible then
				return
			end
			section.hidden = not visible
			visibilityToken += 1
			local token = visibilityToken
			ui(function()
				spr.stop(holder)
				if instant or static or not frame.Visible then
					settle(visible)
					return
				end
				-- AbsoluteSize is in screen pixels; the window may be mid-scale.
				local scale = windowScale.Scale > 0 and windowScale.Scale or 1
				local height = holder.Visible and holder.AbsoluteSize.Y / scale or 0
				holder.AutomaticSize = Enum.AutomaticSize.None
				holder.ClipsDescendants = true
				holder.Size = UDim2.new(1, 0, 0, height)
				holder.Visible = true
				if not visible then
					local hide = entry("section", "hide")
					spr.target(holder, hide.damping, hide.frequency, { Size = UDim2.new(1, 0, 0, 0) })
					spr.completed(holder, function()
						if visibilityToken == token then
							holder.Visible = false
						end
					end)
					return
				end
				task.spawn(function()
					RunService.Heartbeat:Wait() -- let the card lay out before measuring it
					if visibilityToken ~= token then
						return
					end
					local show = entry("section", "show")
					spr.target(holder, show.damping, show.frequency, {
						Size = UDim2.new(1, 0, 0, sectionFrame.AbsoluteSize.Y / scale + 2 + SECTION_GAP),
					})
					spr.completed(holder, function()
						if visibilityToken == token then
							settle(true)
						end
					end)
				end)
			end)
		end

		--[=[ section:addToggle({ title, description, default, callback, id, flag, order }) ]=]
		function section:addToggle(config)
			local id, order, flag
			config, id, order, flag = prepare(config, "Toggle")
			local value = config.default == true
			local row = rowBase(theme, "TextButton", id, order, 48, config.title, config.description, 120, {
				Control = "toggle",
				Value = tostring(value),
			})
			local track = make("Frame", {
				Name = "Track",
				Parent = row,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.new(1, -20, 0.5, 0),
				BackgroundColor3 = theme.control,
				Size = UDim2.fromOffset(40, 22),
				Attr = { PlopacleNode = "toggle-track" },
			}, { corner(0, 6) })
			local knob = make("Frame", {
				Name = "Knob",
				Parent = track,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.new(0, 11, 0.5, 0),
				BackgroundColor3 = theme.knobOff,
				Size = UDim2.fromOffset(16, 16),
				Attr = { PlopacleNode = "toggle-knob" },
			}, { corner(0, 4) })
			remember(knob, "toggle-knob", "press")

			local function apply(instant)
				local state = value and "on" or "off"
				toGoal(track, "toggle-track", state, instant)
				toGoal(knob, "toggle-knob", state, instant)
			end
			apply(true)

			local object = valueObject(rowObject(row, register(row, config)), config, flag, function()
				return value
			end, function(newValue)
				newValue = newValue == true
				if newValue == value then
					return false
				end
				value = newValue
				ui(function()
					row:SetAttribute("Value", tostring(value))
					apply(false)
				end)
				return true
			end)

			row.MouseButton1Down:Connect(function()
				toGoal(knob, "toggle-knob", "press")
			end)
			row.MouseButton1Up:Connect(function()
				toRest(knob, "toggle-knob", "press")
			end)
			row.MouseLeave:Connect(function()
				toRest(knob, "toggle-knob", "press")
			end)
			row.MouseButton1Click:Connect(function()
				object:set(not value)
			end)
			return object
		end

		--[=[ section:addSlider({ title, min, max, step, default, suffix, callback, ... }) ]=]
		function section:addSlider(config)
			local id, order, flag
			config, id, order, flag = prepare(config, "Slider")
			local min = config.min or 0
			local max = config.max or 100
			local step = config.step or 1
			local suffix = config.suffix or ""
			local value = math.clamp(config.default or min, min, max)

			local attributes = { Control = "slider", Value = tostring(value), Min = tostring(min), Max = tostring(max), Step = tostring(step) }
			if suffix ~= "" then
				attributes.Suffix = suffix
			end
			local row = rowBase(theme, "Frame", id, order, 56, config.title, nil, 120, attributes)
			local label = make("TextLabel", {
				Name = "Value",
				Parent = row,
				FontFace = font("Bold"),
				TextColor3 = theme.accent,
				Text = "",
				AnchorPoint = Vector2.new(1, 0),
				BackgroundTransparency = 1,
				TextXAlignment = Enum.TextXAlignment.Right,
				Position = UDim2.new(1, 0, 0, 6),
				TextSize = 14,
				Size = UDim2.fromOffset(112, 18),
			})
			local track = make("TextButton", {
				Name = "Track",
				Parent = row,
				AnchorPoint = Vector2.new(0.5, 0.5),
				BackgroundTransparency = 1,
				Position = UDim2.new(0.5, 0, 0, 40),
				AutoButtonColor = false,
				Text = "",
				Size = UDim2.new(1, 0, 0, 20),
			})
			make("Frame", {
				Name = "Rail",
				Parent = track,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.fromScale(0.5, 0.5),
				BackgroundColor3 = theme.control,
				Size = UDim2.new(1, 0, 0, 4),
			}, { fullCorner() })
			local fill = make("Frame", {
				Name = "Fill",
				Parent = track,
				AnchorPoint = Vector2.new(0, 0.5),
				Position = UDim2.new(0, 0, 0.5, 0),
				BackgroundColor3 = theme.white,
				Size = UDim2.new(0, 0, 0, 4),
				Attr = { PlopacleNode = "slider-fill" },
			}, { fullCorner(), ember(theme) })
			local knob = make("Frame", {
				Name = "Knob",
				Parent = track,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.new(0, 0, 0.5, 0),
				BackgroundColor3 = theme.text,
				Size = UDim2.fromOffset(8, 18),
				Attr = { PlopacleNode = "slider-knob" },
			}, { corner(0, 3) })
			remember(knob, "slider-knob", "hover")

			local function render(instant)
				local alpha = max > min and math.floor((value - min) / (max - min) * 1000 + 0.5) / 1000 or 0
				local fillSize = UDim2.new(alpha, 0, 0, 4)
				local knobPosition = UDim2.new(alpha, 0, 0.5, 0)
				label.Text = formatNumber(value, step) .. suffix
				local fillEntry, knobEntry = entry("slider-fill", "drag"), entry("slider-knob", "drag")
				if instant then
					fill.Size = fillSize
					knob.Position = knobPosition
				else
					spr.target(fill, fillEntry.damping, fillEntry.frequency, { Size = fillSize })
					spr.target(knob, knobEntry.damping, knobEntry.frequency, { Position = knobPosition })
				end
			end
			render(true)

			local object = valueObject(rowObject(row, register(row, config)), config, flag, function()
				return value
			end, function(newValue)
				newValue = tonumber(newValue)
				if not newValue then
					return false
				end
				local stepped = math.clamp(min + math.floor((newValue - min) / step + 0.5) * step, min, max)
				if stepped == value then
					return false
				end
				value = stepped
				ui(function()
					row:SetAttribute("Value", tostring(value))
					render(false)
				end)
				return true
			end)

			local slider = {}
			function slider.setFromX(x)
				local alpha = math.clamp((x - track.AbsolutePosition.X) / math.max(track.AbsoluteSize.X, 1), 0, 1)
				object:set(min + alpha * (max - min))
			end
			function slider.release()
				toRest(knob, "slider-knob", "hover")
			end

			track.InputBegan:Connect(function(input)
				if isPointerDown(input) then
					draggingSlider = slider
					toGoal(knob, "slider-knob", "hover")
					slider.setFromX(input.Position.X)
				end
			end)
			track.MouseEnter:Connect(function()
				toGoal(knob, "slider-knob", "hover")
			end)
			track.MouseLeave:Connect(function()
				if draggingSlider ~= slider then
					toRest(knob, "slider-knob", "hover")
				end
			end)
			return object
		end

		--[=[ section:addButton({ title, description, text, callback, notify = { title, message } }) ]=]
		function section:addButton(config)
			local id, order
			config, id, order = prepare(config, "Button")
			local attributes = { Control = "button" }
			local toast = config.notify
			if toast then
				attributes.Toast = tostring(toast.title or "") .. "|" .. tostring(toast.message or "")
			end
			local row = rowBase(theme, "Frame", id, order, 48, config.title, config.description, 120, attributes)
			local button = make("TextButton", {
				Name = "Button",
				Parent = row,
				FontFace = font("Bold"),
				TextColor3 = theme.accent,
				Text = config.text or "Run",
				AutoButtonColor = false,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.new(1, -40, 0.5, 0),
				BackgroundColor3 = theme.field,
				TextSize = 14,
				Size = UDim2.fromOffset(80, 30),
				Attr = { PlopacleNode = "button" },
			}, {
				corner(0, 6),
				make("UIStroke", {
					Color = theme.accent,
					Transparency = 0.55,
					ApplyStrokeMode = Enum.ApplyStrokeMode.Border,
				}),
			})
			remember(button, "button", "hover")
			remember(button, "button", "press")

			local object = rowObject(row, register(row, config))

			function object:setText(text)
				ui(function()
					button.Text = tostring(text)
				end)
			end

			-- Runs the button's action as if it had been clicked.
			function object:press()
				fire(config.callback)
				if toast then
					window:notify(toast.title, toast.message, toast.duration)
				end
			end

			-- A disabled button is a placeholder: dimmed, and it does nothing.
			if config.disabled then
				button.Active = false
				button.TextColor3 = theme.textDim
				local outline = button:FindFirstChildOfClass("UIStroke")
				outline.Color = theme.border
				outline.Transparency = 0.6
				return object
			end

			button.MouseEnter:Connect(function()
				toGoal(button, "button", "hover")
			end)
			button.MouseLeave:Connect(function()
				toRest(button, "button", "hover")
				toRest(button, "button", "press")
			end)
			button.MouseButton1Down:Connect(function()
				toGoal(button, "button", "press")
			end)
			button.MouseButton1Up:Connect(function()
				toRest(button, "button", "press")
			end)
			button.MouseButton1Click:Connect(function()
				object:press()
			end)
			return object
		end

		--[=[ section:addAction({ text, style = "default" | "primary", callback }): a full-width button ]=]
		function section:addAction(config)
			local id, order
			config, id, order = prepare(config, "Action")
			config.title = config.title or config.text
			local primary = config.style == "primary"
			local row = make("Frame", {
				Name = "Row_" .. id,
				LayoutOrder = order,
				BackgroundTransparency = 1,
				Size = UDim2.new(1, 0, 0, 44),
				Attr = { Control = "action" },
			})
			-- Search and "Go to page" read a row's Title; keep one, invisible.
			make("TextLabel", { Name = "Title", Parent = row, Visible = false, Text = config.title or "" })
			local button = make("TextButton", {
				Name = "Button",
				Parent = row,
				FontFace = font("Bold"),
				TextColor3 = primary and theme.bg or theme.accent,
				Text = config.text or "Run",
				AutoButtonColor = false,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.fromScale(0.5, 0.5),
				BackgroundColor3 = primary and theme.accent or theme.field,
				TextSize = 15,
				Size = UDim2.new(1, 0, 0, 36),
				Attr = { PlopacleNode = primary and "action-primary" or "action" },
			}, { corner(0, 6) })
			if not primary then
				make("UIStroke", {
					Parent = button,
					Color = theme.accent,
					Transparency = 0.55,
					ApplyStrokeMode = Enum.ApplyStrokeMode.Border,
				})
			end
			bindHover(button, primary and "action-primary" or "action")

			local object = rowObject(row, register(row, config))

			function object:setText(text)
				ui(function()
					button.Text = tostring(text)
				end)
			end

			function object:press()
				fire(config.callback)
			end

			button.MouseButton1Click:Connect(function()
				object:press()
			end)
			return object
		end

		--[=[ section:addDropdown({ title, description, options = { ... }, default, callback }) ]=]
		function section:addDropdown(config)
			local id, order, flag
			config, id, order, flag = prepare(config, "Dropdown")
			local optionList = table.clone(config.options or {})
			local value = config.default
			if value == nil then
				value = optionList[1]
			end

			local labels = {}
			for index, option in ipairs(optionList) do
				labels[index] = tostring(option)
			end
			local row = rowBase(theme, "Frame", id, order, 48, config.title, config.description, 184, {
				Control = "dropdown",
				Value = tostring(value),
				Options = table.concat(labels, "|"),
			})
			local button = make("TextButton", {
				Name = "Dropdown",
				Parent = row,
				AnchorPoint = Vector2.new(0.5, 0.5),
				BackgroundColor3 = theme.field,
				Position = UDim2.new(1, -80, 0.5, 0),
				ClipsDescendants = true,
				AutoButtonColor = false,
				Text = "",
				Size = UDim2.fromOffset(160, 30),
				Attr = { PlopacleNode = "dropdown" },
			}, { corner(0, 6) })
			local stroke = make("UIStroke", {
				Parent = button,
				Color = theme.border,
				Transparency = 0.4,
				ApplyStrokeMode = Enum.ApplyStrokeMode.Border,
				Attr = { PlopacleNode = "dropdown-stroke" },
			})
			local label = make("TextLabel", {
				Name = "Value",
				Parent = button,
				FontFace = font("Medium"),
				TextColor3 = theme.text,
				Text = tostring(value),
				Position = UDim2.fromOffset(12, 0),
				BackgroundTransparency = 1,
				TextXAlignment = Enum.TextXAlignment.Left,
				TextTruncate = Enum.TextTruncate.AtEnd,
				TextSize = 14,
				Size = UDim2.new(1, -38, 1, 0),
				Attr = { PlopacleNode = "dropdown-value" },
			})
			local chevron = make("TextLabel", {
				Name = "Chevron",
				Parent = button,
				FontFace = font("Regular"),
				TextColor3 = theme.textDim,
				Text = "\u{25BC}",
				AnchorPoint = Vector2.new(0.5, 0.5),
				BackgroundTransparency = 1,
				Position = UDim2.new(1, -16, 0.5, 0),
				TextSize = 9,
				Size = UDim2.fromOffset(12, 12),
				Attr = { PlopacleNode = "dropdown-chevron" },
			})

			local dropdown = { button = button, chevron = chevron, stroke = stroke, options = optionList, search = config.search }
			if config._multi then
				dropdown.isOn, dropdown.toggle = config._multi.isOn, config._multi.toggle
			end

			local object = valueObject(rowObject(row, register(row, config)), config, flag, function()
				return value
			end, function(newValue)
				if newValue == value or table.find(optionList, newValue) == nil then
					return false
				end
				value = newValue
				ui(function()
					row:SetAttribute("Value", tostring(value))
					label.Text = tostring(value)
					playFrom(label, "dropdown-value", "change")
				end)
				return true
			end)

			function dropdown.get()
				return value
			end

			function dropdown.set(option)
				object:set(option)
			end

			-- Replace the choices. `selected`, if offered, becomes the value without running the
			-- callback; otherwise the current value is kept if it is still offered.
			function object:setOptions(newOptions, selected)
				if openMenu == dropdown then
					closeDropdown()
				end
				table.clear(optionList)
				for _, option in ipairs(newOptions) do
					table.insert(optionList, option)
				end
				if selected ~= nil and table.find(optionList, selected) ~= nil then
					object:set(selected, true)
				elseif table.find(optionList, value) == nil then
					value = nil
					object:set(optionList[1])
				end
			end

			bindHover(button, "dropdown")
			button.MouseButton1Click:Connect(function()
				openDropdown(dropdown)
			end)
			return object
		end

		--[=[ section:addMultiDropdown({ title, description, items = { { id, title, default, callback }, ... }, summary })
			A dropdown whose options are toggles. Each item is a flagged control of its own (saved
			with the config) and the menu stays open while they are ticked. `summary(titles)` may
			return the text for the field; by default "All", "None" or "2 of 4".
			Returns an object with get(id), set(id, on, silent) and items[id] = { get, set }. ]=]
		function section:addMultiDropdown(config)
			config = config or {}
			local entries, titles, byTitle, byId = {}, {}, {}, {}
			for index, itemConfig in ipairs(config.items or {}) do
				local name = tostring(itemConfig.title or itemConfig.id or index)
				local item = {
					id = itemConfig.id or identifier(name),
					title = name,
					on = itemConfig.default == true,
					callback = itemConfig.callback,
					flag = itemConfig.flag,
				}
				if item.flag == nil then
					item.flag = page.name .. "." .. item.id
				end
				table.insert(entries, item)
				table.insert(titles, name)
				byTitle[name], byId[item.id] = item, item
			end

			local object
			local function refresh()
				local picked = {}
				for _, item in ipairs(entries) do
					if item.on then
						table.insert(picked, item.title)
					end
				end
				local text
				if type(config.summary) == "function" then
					local ok, result = pcall(config.summary, picked)
					text = ok and tostring(result) or nil
				end
				text = text or (#picked == #entries and "All" or #picked == 0 and "None"
					or ("%d of %d"):format(#picked, #entries))
				ui(function()
					local field = object.instance:FindFirstChild("Dropdown")
					local label = field and field:FindFirstChild("Value")
					if label then
						label.Text = text
					end
					object.instance:SetAttribute("Value", text)
				end)
			end

			local function setItem(item, on, silent)
				on = on == true
				if item.on == on then
					return false
				end
				item.on = on
				refresh()
				if not silent then
					fire(item.callback, on)
					valueChanged()
				end
				return true
			end

			object = section:addDropdown({
				id = config.id,
				title = config.title,
				description = config.description,
				search = config.search,
				options = titles,
				default = titles[1],
				flag = false,
				_multi = {
					isOn = function(title)
						return byTitle[title] ~= nil and byTitle[title].on
					end,
					toggle = function(title)
						local item = byTitle[title]
						if item then
							setItem(item, not item.on)
						end
					end,
				},
			})
			object.items = {}
			for _, item in ipairs(entries) do
				object.items[item.id] = {
					get = function()
						return item.on
					end,
					set = function(_, on, silent)
						setItem(item, on, silent)
					end,
				}
				if item.flag ~= false then
					window.flags[item.flag] = {
						get = function()
							return item.on
						end,
						set = function(value)
							setItem(item, value)
						end,
					}
				end
			end
			function object:get(id)
				return byId[id] ~= nil and byId[id].on
			end
			function object:set(id, on, silent)
				if byId[id] then
					setItem(byId[id], on, silent)
				end
			end
			refresh()
			return object
		end

		--[=[ section:addKeybind({ title, description, default, callback, onPressed }) ]=]
		function section:addKeybind(config)
			local id, order, flag
			config, id, order, flag = prepare(config, "Keybind")
			local key = parseKey(config.default)
			local isToggleKey = config._toggleKey == true

			local attributes = { Control = "keybind", Value = key and key.Name or "None" }
			if isToggleKey then
				attributes.Role = "ToggleKey"
			end
			local row = rowBase(theme, "Frame", id, order, 48, config.title, config.description, 120, attributes)
			local button = make("TextButton", {
				Name = "Key",
				Parent = row,
				FontFace = font("Bold"),
				TextColor3 = theme.text,
				Text = keyName(key),
				AutoButtonColor = false,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.new(1, -40, 0.5, 0),
				BackgroundColor3 = theme.field,
				TextSize = 14,
				Size = UDim2.fromOffset(80, 30),
				Attr = { PlopacleNode = "keybind" },
			}, { corner(0, 6) })
			local stroke = make("UIStroke", {
				Name = "Stroke",
				Parent = button,
				Color = theme.border,
				Transparency = 0.4,
				ApplyStrokeMode = Enum.ApplyStrokeMode.Border,
				Attr = { PlopacleNode = "keybind-stroke" },
			})
			remember(stroke, "keybind-stroke", "listening")

			local bind = { onPressed = config.onPressed }

			local function show()
				button.Text = keyName(key)
				if isToggleKey then
					setToggleKey(key)
				end
			end
			show()

			local object = valueObject(rowObject(row, register(row, config)), config, flag, function()
				return key
			end, function(newValue)
				local code = parseKey(newValue)
				if code == key or (code == nil and isToggleKey) then
					return false
				end
				key = code
				ui(function()
					row:SetAttribute("Value", key and key.Name or "None")
					show()
				end)
				return true
			end)

			function bind.key()
				return key
			end

			function bind.cancel()
				if listeningBind == bind then
					listeningBind = nil
				end
				toRest(stroke, "keybind-stroke", "listening")
				show()
			end

			-- Escape keeps the current key, Backspace clears it (never for the menu key: that
			-- would leave no way to bring the window back), anything else becomes the new key.
			function bind.capture(code)
				if code == Enum.KeyCode.Backspace then
					object:set(nil)
				elseif code ~= Enum.KeyCode.Escape and code ~= Enum.KeyCode.Unknown then
					object:set(code)
				end
				bind.cancel()
			end

			-- True while the bound key is held.
			function object:isDown()
				return key ~= nil and UserInputService:IsKeyDown(key)
			end

			if not isToggleKey then
				table.insert(keybinds, bind)
			end
			bindHover(button, "keybind")
			button.MouseButton1Click:Connect(function()
				if listeningBind == bind then
					bind.cancel()
					return
				end
				if listeningBind then
					listeningBind.cancel()
				end
				listeningBind = bind
				button.Text = "Press a key"
				toGoal(stroke, "keybind-stroke", "listening")
			end)
			return object
		end

		--[=[ section:addInput({ title, description, placeholder, default, numeric, callback }) ]=]
		function section:addInput(config)
			local id, order, flag
			config, id, order, flag = prepare(config, "Input")
			local value = config.default ~= nil and tostring(config.default) or ""
			local row = rowBase(theme, "Frame", id, order, 48, config.title, config.description, 224, {
				Control = "input",
				Value = value,
			})
			local field = make("Frame", {
				Name = "Field",
				Parent = row,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.new(1, -100, 0.5, 0),
				BackgroundColor3 = theme.bg,
				Size = UDim2.fromOffset(200, 30),
			}, { corner(0, 6) })
			local stroke = make("UIStroke", {
				Name = "Stroke",
				Parent = field,
				Color = theme.border,
				Transparency = 0.4,
				Attr = { PlopacleNode = "field-stroke" },
			})
			local box = make("TextBox", {
				Name = "Input",
				Parent = field,
				FontFace = font("Regular"),
				ClipsDescendants = true,
				TextColor3 = theme.text,
				Text = value,
				ClearTextOnFocus = false,
				Position = UDim2.fromOffset(10, 0),
				TextXAlignment = Enum.TextXAlignment.Left,
				BackgroundTransparency = 1,
				PlaceholderColor3 = theme.textDim,
				TextSize = 14,
				PlaceholderText = config.placeholder or "",
				Size = UDim2.new(1, -20, 1, 0),
			})
			remember(stroke, "field-stroke", "focus")

			local object = valueObject(rowObject(row, register(row, config)), config, flag, function()
				if config.numeric then
					return tonumber(value)
				end
				return value
			end, function(newValue)
				newValue = newValue ~= nil and tostring(newValue) or ""
				if config.numeric and newValue ~= "" and tonumber(newValue) == nil then
					ui(function()
						box.Text = value
					end)
					return false
				end
				if newValue == value then
					return false
				end
				value = newValue
				ui(function()
					row:SetAttribute("Value", value)
					box.Text = value
				end)
				return true
			end)

			box.Focused:Connect(function()
				toGoal(stroke, "field-stroke", "focus")
			end)
			box.FocusLost:Connect(function()
				toRest(stroke, "field-stroke", "focus")
				object:set(box.Text)
			end)
			return object
		end

		--[=[ section:addLabel({ title, description }) ]=]
		function section:addLabel(config)
			if type(config) == "string" then
				config = { title = config }
			end
			local id, order
			config, id, order = prepare(config, "Label")
			local hasDescription = config.description ~= nil and config.description ~= ""
			local row = rowBase(theme, "Frame", id, order, hasDescription and 48 or 30, config.title, config.description, 0, { Control = "label" })
			return rowObject(row, register(row, config))
		end

		--[=[ section:addParagraph(text): wrapped body text ]=]
		function section:addParagraph(config)
			if type(config) == "string" then
				config = { text = config }
			end
			local id, order
			config, id, order = prepare(config, "Paragraph")
			config.title = config.title or config.text
			local row = make("Frame", {
				Name = "Row_" .. id,
				LayoutOrder = order,
				BackgroundTransparency = 1,
				AutomaticSize = Enum.AutomaticSize.Y,
				Size = UDim2.new(1, 0, 0, 0),
				Attr = { Control = "paragraph" },
			}, {
				make("UIPadding", { PaddingTop = UDim.new(0, 4), PaddingBottom = UDim.new(0, 6) }),
			})
			local body = make("TextLabel", {
				Name = "Title",
				Parent = row,
				FontFace = font("Regular"),
				TextColor3 = theme.textDim,
				Text = config.text or "",
				TextWrapped = true,
				BackgroundTransparency = 1,
				TextXAlignment = Enum.TextXAlignment.Left,
				TextYAlignment = Enum.TextYAlignment.Top,
				TextSize = 13,
				LineHeight = 1.15,
				AutomaticSize = Enum.AutomaticSize.Y,
				Size = UDim2.new(1, 0, 0, 0),
			})
			local object = rowObject(row, register(row, config))

			function object:set(text)
				ui(function()
					body.Text = tostring(text)
				end)
			end
			return object
		end

		--[=[ section:addStat({ title, value, tone }): a label with a live value on the right ]=]
		function section:addStat(config)
			local id, order
			config, id, order = prepare(config, "Stat")
			local row = rowBase(theme, "Frame", id, order, 30, config.title, nil, 150, { Control = "stat" })
			local label = make("TextLabel", {
				Name = "Value",
				Parent = row,
				FontFace = font("Bold"),
				TextColor3 = theme.text,
				Text = config.value ~= nil and tostring(config.value) or "\u{2013}",
				AnchorPoint = Vector2.new(1, 0),
				BackgroundTransparency = 1,
				TextXAlignment = Enum.TextXAlignment.Right,
				TextTruncate = Enum.TextTruncate.AtEnd,
				Position = UDim2.new(1, 0, 0, 6),
				TextSize = 15,
				Size = UDim2.fromOffset(150, 18),
			})
			local object = rowObject(row, register(row, config))

			-- set(value, tone): tone is "text" (default), "accent", "good", "warn", "bad" or "dim".
			function object:set(value, tone)
				ui(function()
					label.Text = tostring(value)
					if tone then
						label.TextColor3 = theme[tone == "dim" and "textDim" or tone] or theme.text
					end
				end)
			end
			if config.tone then
				object:set(label.Text, config.tone)
			end
			return object
		end

		--[=[ section:addStats({ columns = 2, items = { { id, title, value }, ... } }) -> { [id] = stat } ]=]
		function section:addStats(config)
			local id, order
			config, id, order = prepare(config, "Stats")
			local columns = math.max(config.columns or 2, 1)
			local gap = 8
			local row = make("Frame", {
				Name = "Row_" .. id,
				LayoutOrder = order,
				BackgroundTransparency = 1,
				AutomaticSize = Enum.AutomaticSize.Y,
				Size = UDim2.new(1, 0, 0, 0),
				Attr = { Control = "stats" },
			}, {
				make("UIPadding", { PaddingTop = UDim.new(0, 4), PaddingBottom = UDim.new(0, 8) }),
				make("UIGridLayout", {
					CellSize = UDim2.new(1 / columns, -math.ceil(gap * (columns - 1) / columns), 0, 58),
					CellPadding = UDim2.fromOffset(gap, gap),
					SortOrder = Enum.SortOrder.LayoutOrder,
				}),
			})
			local titles = {}
			local tiles = {}
			for index, itemConfig in ipairs(config.items or {}) do
				table.insert(titles, tostring(itemConfig.title or ""))
				local tile = make("Frame", {
					Name = "Tile_" .. (itemConfig.id or identifier(itemConfig.title or index)),
					Parent = row,
					LayoutOrder = index,
					BackgroundColor3 = theme.field,
				}, { corner(0, 6) })
				make("TextLabel", {
					Name = "Title",
					Parent = tile,
					FontFace = font("Regular"),
					TextColor3 = theme.textDim,
					Text = tostring(itemConfig.title or ""),
					BackgroundTransparency = 1,
					TextXAlignment = Enum.TextXAlignment.Left,
					TextTruncate = Enum.TextTruncate.AtEnd,
					Position = UDim2.fromOffset(12, 9),
					TextSize = 13,
					Size = UDim2.new(1, -24, 0, 16),
				})
				local valueLabel = make("TextLabel", {
					Name = "Value",
					Parent = tile,
					FontFace = font("Bold"),
					TextColor3 = theme.text,
					Text = itemConfig.value ~= nil and tostring(itemConfig.value) or "\u{2013}",
					BackgroundTransparency = 1,
					TextXAlignment = Enum.TextXAlignment.Left,
					TextTruncate = Enum.TextTruncate.AtEnd,
					Position = UDim2.fromOffset(12, 28),
					TextSize = 19,
					Size = UDim2.new(1, -24, 0, 22),
				})
				local tileObject = { instance = tile }
				function tileObject:set(value, tone)
					ui(function()
						valueLabel.Text = tostring(value)
						if tone then
							valueLabel.TextColor3 = theme[tone == "dim" and "textDim" or tone] or theme.text
						end
					end)
				end
				tiles[itemConfig.id or identifier(itemConfig.title or index)] = tileObject
			end
			-- Search reads a row's Title; keep one, invisible, listing every tile.
			config.title = config.title or table.concat(titles, " ")
			make("TextLabel", {
				Name = "Title",
				Visible = false,
				Text = config.title,
				Parent = make("Folder", { Name = "Meta", Parent = row }),
			})
			local object = rowObject(row, register(row, config))
			object.setTitle = nil
			for key, tileObject in pairs(tiles) do
				object[key] = tileObject
			end
			object.tiles = tiles
			return object
		end

		--[=[ section:addProgress({ title, value = 0..1, text, tone }): a labelled bar ]=]
		function section:addProgress(config)
			local id, order
			config, id, order = prepare(config, "Progress")
			local value = math.clamp(config.value or 0, 0, 1)
			local row = rowBase(theme, "Frame", id, order, 44, config.title, nil, 120, { Control = "progress" })
			local label = make("TextLabel", {
				Name = "Value",
				Parent = row,
				FontFace = font("Bold"),
				TextColor3 = theme.text,
				Text = config.text or ("%d%%"):format(math.floor(value * 100 + 0.5)),
				AnchorPoint = Vector2.new(1, 0),
				BackgroundTransparency = 1,
				TextXAlignment = Enum.TextXAlignment.Right,
				Position = UDim2.new(1, 0, 0, 6),
				TextSize = 14,
				Size = UDim2.fromOffset(112, 18),
			})
			local rail = make("Frame", {
				Name = "Rail",
				Parent = row,
				AnchorPoint = Vector2.new(0.5, 0.5),
				Position = UDim2.new(0.5, 0, 0, 35),
				BackgroundColor3 = theme.control,
				Size = UDim2.new(1, 0, 0, 6),
			}, { fullCorner() })
			local fill = make("Frame", {
				Name = "Fill",
				Parent = rail,
				BackgroundColor3 = theme.white,
				Size = UDim2.fromScale(value, 1),
			}, { fullCorner() })
			local gradient = ember(theme)
			gradient.Parent = fill

			local object = rowObject(row, register(row, config))

			local function setTone(tone)
				if tone == nil or tone == "accent" then
					gradient.Color = ColorSequence.new(theme.accent, theme.accent2)
				else
					local colour = theme[tone] or theme.accent
					gradient.Color = ColorSequence.new(colour, colour)
				end
			end
			setTone(config.tone)

			-- set(value 0..1, text, tone). Text defaults to a percentage.
			function object:set(newValue, text, tone)
				value = math.clamp(tonumber(newValue) or 0, 0, 1)
				ui(function()
					label.Text = text ~= nil and tostring(text) or ("%d%%"):format(math.floor(value * 100 + 0.5))
					if tone then
						setTone(tone)
					end
					local drag = entry("slider-fill", "drag")
					spr.target(fill, drag.damping, drag.frequency, { Size = UDim2.fromScale(value, 1) })
				end)
			end

			function object:get()
				return value
			end

			function object:setTone(tone)
				ui(function()
					setTone(tone)
				end)
			end
			return object
		end

		return section
	end

	--[=[ window:addPage({ name, icon, order }) -> page ]=]
	function window:addPage(config)
		if type(config) == "string" then
			config = { name = config }
		end
		local name = config.name or ("Page" .. (pageOrder + 1))
		if pages[name] then
			return pages[name]
		end

		local order = config.order
		if not order then
			pageOrder += 1
			order = pageOrder
		end
		local tab = buildTab(theme, name, config.icon or "Folder", order)
		tab.button.Parent = refs.tabList
		-- Sections carry their own gap (see addSection), so the list adds none and the page's
		-- bottom padding gives up what the last section's gap already provides.
		local instance = scroller(theme, "Page_" .. identifier(name), { Page = name }, 0)
		instance:FindFirstChildOfClass("UIPadding").PaddingBottom = UDim.new(0, 24 - SECTION_GAP)
		instance.Parent = pageHost

		local page = { name = name, instance = instance, tab = tab, sections = {}, sectionOrder = 0 }
		pages[name] = page

		--[=[ page:addSection(title | { title, order }) -> section ]=]
		function page:addSection(sectionConfig)
			return addSection(page, sectionConfig)
		end

		function page:select()
			window:selectPage(name)
		end

		tab.button.MouseButton1Click:Connect(function()
			window:selectPage(name)
		end)
		tab.button.MouseEnter:Connect(function()
			if currentPage ~= name then
				toGoal(tab.button, "tab", "hover")
			end
		end)
		tab.button.MouseLeave:Connect(function()
			toGoal(tab.button, "tab", currentPage == name and "active" or "inactive")
		end)

		-- The first page added opens; a pinned page (Settings) never takes that slot.
		if not currentPage and not config._pinned then
			selectPage(name, true)
		end
		return page
	end

	-----------------------------------------------------------------------------------------
	-- Window-level setters

	function window:setTitle(text)
		ui(function()
			refs.brandTitle.Text = tostring(text)
		end)
	end

	function window:setSubtitle(text)
		ui(function()
			refs.caption.Text = tostring(text)
		end)
	end

	function window:setToggleKey(code)
		local control = window.flags["Settings.ToggleKey"]
		if control then
			control.set(code)
		else
			setToggleKey(parseKey(code))
		end
	end

	--[=[ window:get(flag) / window:set(flag, value): read or write any control by its flag ]=]
	function window:get(flag)
		local control = window.flags[flag]
		return control and control.get()
	end

	function window:set(flag, value)
		local control = window.flags[flag]
		if control then
			control.set(value)
		end
	end

	function window:onUnload(callback)
		table.insert(unloadCallbacks, callback)
	end

	-- Removes the window, disconnects everything and runs the onUnload callbacks.
	function window:destroy()
		if destroyed then
			return
		end
		destroyed = true
		if menuFollow then
			menuFollow:Disconnect()
			menuFollow = nil
		end
		for _, connection in ipairs(connections) do
			connection:Disconnect()
		end
		table.clear(connections)
		for _, callback in ipairs(unloadCallbacks) do
			fire(callback)
		end
		gui:Destroy()
	end

	-----------------------------------------------------------------------------------------
	-- Built-in Settings page (pinned to the bottom of the sidebar)

	if options.settingsPage ~= false then
		local settings = window:addPage({ name = "Settings", icon = "Sliders", order = 100, _pinned = true })
		window.settings = settings
		local interface = settings:addSection("Interface")
		interface:addKeybind({
			id = "ToggleKey",
			order = 3,
			title = "Menu key",
			description = "Shows and hides this window",
			default = toggleKey,
			_toggleKey = true,
		})
		interface:addButton({
			id = "Unload",
			order = 9,
			title = "Unload hub",
			description = "Stop everything and remove this window",
			text = "Unload",
			callback = function()
				window:destroy()
			end,
		})

		local configs = settings:addSection("Configs")
		if fileApi() then
			local picker = configs:addDropdown({
				id = "Config",
				title = "Config",
				description = "The saved setup in use",
				options = window:listConfigs(),
				default = activeConfig,
				flag = false,
				callback = function(name)
					if name ~= activeConfig and window:loadConfig(name) then
						window:notify("Config loaded", name)
					end
				end,
			})
			configChanged = function()
				picker:setOptions(window:listConfigs(), activeConfig)
			end
			local nameBox = configs:addInput({
				id = "ConfigName",
				title = "Name",
				description = "For a new config, or to rename this one",
				placeholder = "Config name",
				flag = false,
			})
			configs:addButton({
				id = "CreateConfig",
				title = "New config",
				description = "Save the current settings under that name",
				text = "Create",
				callback = function()
					local ok, reason = window:createConfig(nameBox:get())
					if ok then
						nameBox:set("", true)
						window:notify("Config created", activeConfig)
					else
						window:notify("Config not created", reason)
					end
				end,
			})
			configs:addButton({
				id = "RenameConfig",
				title = "Rename config",
				description = "Give the config in use that name",
				text = "Rename",
				callback = function()
					local ok, reason = window:renameConfig(nameBox:get())
					if ok then
						nameBox:set("", true)
						window:notify("Config renamed", activeConfig)
					else
						window:notify("Config not renamed", reason)
					end
				end,
			})
			configs:addButton({
				id = "SaveConfig",
				title = "Save config",
				description = "Write the current settings to the config in use",
				text = "Save",
				callback = function()
					if window:saveConfig() then
						window:notify("Config saved", activeConfig)
					else
						window:notify("Config not saved", "The file could not be written")
					end
				end,
			})
			configs:addButton({
				id = "DeleteConfig",
				title = "Delete config",
				description = "Removes the config in use and goes back to default",
				text = "Delete",
				callback = function()
					local name = activeConfig
					local ok, reason = window:deleteConfig()
					if ok then
						window:notify("Config deleted", name)
					else
						window:notify("Config not deleted", reason)
					end
				end,
			})
		else
			configs:addParagraph("Configs need an executor that can read and write files.")
		end

		-- Placeholders: a config already saves as a self-describing payload (format version,
		-- name, values), which is what an export code will carry.
		local sharing = settings:addSection("Sharing")
		sharing:addButton({
			id = "ExportConfig",
			title = "Export config",
			description = "Copy a code other people can import",
			text = "Soon",
			disabled = true,
		})
		sharing:addButton({
			id = "ImportConfig",
			title = "Import config",
			description = "Load a config from someone's code",
			text = "Soon",
			disabled = true,
		})
	else
		setToggleKey(toggleKey)
	end

	-----------------------------------------------------------------------------------------
	-- Start

	if static then
		gui.Parent = options.parent
		return window
	end

	-- Input: menu key, keybind capture and onPressed, slider and window dragging.
	track(UserInputService.InputBegan:Connect(function(input)
		if input.UserInputType ~= Enum.UserInputType.Keyboard then
			return
		end
		if listeningBind then
			listeningBind.capture(input.KeyCode)
			return
		end
		local typing = UserInputService:GetFocusedTextBox() ~= nil
		if input.KeyCode == toggleKey then
			-- A letter bound as the menu key must not fire while typing in a text box.
			if not (typing and not isModifier(input.KeyCode)) then
				window:toggle()
			end
			return
		end
		if typing then
			return
		end
		for _, bind in ipairs(keybinds) do
			if bind.onPressed and bind.key() == input.KeyCode then
				fire(bind.onPressed, input.KeyCode)
			end
		end
	end))

	local dragOrigin, dragStartPosition
	for _, handle in ipairs({ refs.topBar, refs.brand }) do
		handle.InputBegan:Connect(function(input)
			if isPointerDown(input) then
				dragOrigin = input.Position
				dragStartPosition = frame.Position
			end
		end)
	end

	track(UserInputService.InputChanged:Connect(function(input)
		if not isPointerMove(input) then
			return
		end
		if draggingSlider then
			draggingSlider.setFromX(input.Position.X)
		elseif dragOrigin then
			local delta = input.Position - dragOrigin
			frame.Position = UDim2.new(
				dragStartPosition.X.Scale,
				dragStartPosition.X.Offset + delta.X,
				dragStartPosition.Y.Scale,
				dragStartPosition.Y.Offset + delta.Y
			)
		end
	end))

	track(UserInputService.InputEnded:Connect(function(input)
		if not isPointerDown(input) then
			return
		end
		if draggingSlider then
			draggingSlider.release()
			draggingSlider = nil
		end
		if dragOrigin then
			dragOrigin = nil
			positionAsFraction()
			snapToPixels()
		end
	end))

	searchInput:GetPropertyChangedSignal("Text"):Connect(function()
		runSearch(searchInput.Text)
	end)
	remember(refs.searchBar, "search-bar", "focus")
	remember(refs.searchStroke, "search-stroke", "focus")
	searchInput.Focused:Connect(function()
		toGoal(refs.searchBar, "search-bar", "focus")
		toGoal(refs.searchStroke, "search-stroke", "focus")
	end)
	searchInput.FocusLost:Connect(function()
		toRest(refs.searchBar, "search-bar", "focus")
		toRest(refs.searchStroke, "search-stroke", "focus")
	end)

	-- A second run of the same script replaces the first window instead of stacking on it.
	local parent = resolveParent(options)
	local marker = "Plorch:" .. (options.name or options.title or "Plorch")
	for _, child in ipairs(parent:GetChildren()) do
		if child:GetAttribute("PlorchWindow") == marker then
			child:Destroy()
		end
	end
	gui:SetAttribute("PlorchWindow", marker)

	frame.Visible = false
	gui.Parent = parent

	local camera = workspace.CurrentCamera
	if camera then
		track(camera:GetPropertyChangedSignal("ViewportSize"):Connect(function()
			local was = fit
			fit = measureFit()
			if fit ~= was then
				-- Keep whatever the open or close spring is doing, at the new size.
				local settled = isOpen and fit or entry("window-scale", "close").goal.Scale * fit
				spr.stop(windowScale)
				windowScale.Scale = settled
			end
			if not dragOrigin then
				-- Drop the last nudge; it belonged to the old viewport.
				frame.Position = UDim2.fromScale(frame.Position.X.Scale, frame.Position.Y.Scale)
			end
			task.defer(snapToPixels)
		end))
	end

	-- Touch devices have no menu key: a small button opens and closes the window. It can be
	-- dragged out of the way, and a touch that does not move is a tap.
	local wantsButton = options.mobileButton
	if wantsButton == nil then
		wantsButton = UserInputService.TouchEnabled and not UserInputService.KeyboardEnabled
	end
	if wantsButton then
		local inset = GuiService:GetGuiInset()
		local button = make("TextButton", {
			Name = "MobileToggle",
			Parent = gui,
			AnchorPoint = Vector2.new(0.5, 0),
			Position = UDim2.new(0.5, 0, 0, inset.Y + 8),
			Size = UDim2.fromOffset(MOBILE_BUTTON_SIZE, MOBILE_BUTTON_SIZE),
			BackgroundColor3 = theme.white,
			AutoButtonColor = false,
			FontFace = font("ExtraBold"),
			TextColor3 = theme.bg,
			Text = refs.logo.Text,
			TextSize = 22,
			ZIndex = 50,
		}, { corner(0, 12), ember(theme, 45) })
		refs.mobileButton = button

		local pressAt, pressPosition, moved
		button.InputBegan:Connect(function(input)
			if isPointerDown(input) then
				pressAt, pressPosition, moved = input.Position, button.Position, false
			end
		end)
		track(UserInputService.InputChanged:Connect(function(input)
			if not (pressAt and isPointerMove(input)) then
				return
			end
			local delta = input.Position - pressAt
			if not moved and delta.Magnitude < TAP_SLOP then
				return
			end
			moved = true
			button.Position = UDim2.new(
				pressPosition.X.Scale,
				pressPosition.X.Offset + delta.X,
				pressPosition.Y.Scale,
				pressPosition.Y.Offset + delta.Y
			)
		end))
		track(UserInputService.InputEnded:Connect(function(input)
			if not (pressAt and isPointerDown(input)) then
				return
			end
			local tapped = not moved
			pressAt = nil
			if tapped then
				window:toggle()
			end
		end))
	end

	-- Open after the caller has added its pages, so the window appears complete.
	task.defer(function()
		if destroyed then
			return
		end
		snapToPixels()
		if currentPage then
			setTitleIcon(pages[currentPage].tab.icon)
		elseif window.settings then
			selectPage("Settings", true)
		end
		if options.startOpen ~= false then
			window:setOpen(true)
		end
	end)

	return window
end

---------------------------------------------------------------------------------------------
-- Reference build

--[=[
	Plorch.create(parent) -> handles

	Rebuilds the Studio original (the Bloxburg sample hub) under `parent` with no behaviour
	running. It exists so Plopacle's `verify_export` can diff this library against the UI it
	was authored from; scripts should use Plorch.new.
]=]
function Plorch.create(parent)
	local window = Plorch.new({
		_static = true,
		parent = parent,
		name = "Hub",
		title = "Plorch",
		subtitle = "Bloxburg",
	})

	local autofarm = window:addPage({ name = "Autofarm", icon = "Replay" })
	local jobs = autofarm:addSection("Jobs")
	jobs:addToggle({ id = "PizzaDelivery", title = "Pizza Delivery", description = "Deliver pizzas on the moped", default = true })
	jobs:addToggle({ id = "IceCream", title = "Ice Cream Seller", description = "Serve customers at the beach stand" })
	jobs:addToggle({ id = "PizzaBaker", title = "Pizza Baker", description = "Work all four baker stations" })
	jobs:addToggle({ id = "GhostPatrol", title = "Ghost Patrol", description = "Catch and deposit ghosts in the cruiser" })

	local movement = autofarm:addSection("Movement")
	movement:addSlider({ id = "RideSpeed", title = "Moped speed", min = 15, max = 62, default = 40, suffix = " studs/s" })
	movement:addSlider({ id = "CruiserSpeed", title = "Cruiser speed", min = 38, max = 85, default = 81, suffix = " studs/s" })
	movement:addToggle({ id = "Underground", title = "Drive underground", description = "Keeps the cruiser out of other players' way" })

	local needs = autofarm:addSection("Needs")
	needs:addToggle({ id = "NeedsBreak", title = "Needs breaks", description = "Go home and refill when a need runs low", default = true })
	needs:addSlider({ id = "BreakThreshold", title = "Break threshold", min = 0, max = 100, step = 5, default = 30, suffix = "%" })
	needs:addButton({
		id = "FixNow",
		title = "Fix needs now",
		description = "Start a needs trip right away",
		text = "Run",
		notify = { title = "Needs trip started", message = "Heading home to refill, then back to work." },
	})
	needs:addDropdown({
		id = "EatAt",
		title = "Eat at",
		description = "Where Hunger gets refilled",
		options = { "Drive-through", "Fridge quick meal", "Don't eat" },
	})

	local session = autofarm:addSection("Session")
	session:addButton({ id = "EndShift", title = "End shift", description = "Stop the running job and clock out", text = "End" })
	session:addInput({
		id = "Webhook",
		title = "Discord webhook",
		description = "Posts a summary when a session ends",
		placeholder = "Paste a webhook URL",
	})

	local player = window:addPage({ name = "Player", icon = "AccountCircle" })
	local character = player:addSection("Character")
	character:addSlider({ id = "WalkSpeed", title = "Walk speed", min = 8, max = 22, default = 15, suffix = " studs/s" })
	character:addToggle({ id = "AntiAfk", title = "Anti-AFK", description = "Stops the idle kick during long sessions", default = true })
	character:addKeybind({ id = "SprintKey", title = "Sprint key", description = "Hold to sprint while walking", default = Enum.KeyCode.LeftShift })
	character:addDropdown({
		id = "MoveMode",
		title = "Movement",
		description = "How the character gets around between tasks",
		options = { "Walk", "Run", "Sprint" },
		default = "Run",
	})

	local interface = window.settings:addSection("Interface")
	interface:addToggle({
		id = "DismissPopups",
		order = 1,
		title = "Dismiss popups",
		description = "Close paycheck and message boxes automatically",
		default = true,
	})
	interface:addButton({
		id = "TestToast",
		order = 4,
		title = "Test notification",
		description = "Send a sample notification",
		text = "Send",
		notify = { title = "Plorch", message = "This is what a notification looks like." },
	})

	local handles = { root = window.instance, window = window }
	for name, instance in pairs(window.refs) do
		handles[name] = instance
	end
	return handles
end

Plorch.icons = ICONS
Plorch.tuning = TUNING

return Plorch
