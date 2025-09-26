"""
Discord Bot Demo for Crypto ML Project

Features:
- !ping: Health check
- !help: Show commands
- !dudoan: Demo price prediction using production quick_loader if available
- !price | !gia [SYMBOL]: Show current price and quick prediction
- !movers: Show top gainers/losers (24h) if data available, else stub
- !chart [SYMBOL]: Show a link to chart (stub) or ascii trend

Token resolution order:
1. BOT_TOKEN environment variable
2. token.txt file at repository root

The bot is safe to run without a token (exits with a helpful message).
"""

import os
import json
import time  # Add time import for cooldown system
import atexit
import platform
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple, Optional
import asyncio
import requests

import discord
from discord.ext import commands, tasks

try:
	import psutil
except ImportError:
	psutil = None


def read_bot_token() -> Optional[str]:
	token = os.getenv("BOT_TOKEN")
	if token:
		return token.strip()
	# Fallback to token.txt in repo root
	root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
	token_file = os.path.join(root, "token.txt")
	if os.path.exists(token_file):
		try:
			with open(token_file, "r", encoding="utf-8") as f:
				line = f.readline().strip()
				return line or None
		except Exception:
			return None
	return None


def try_predict(features: dict) -> dict:
	"""Attempt to predict using production quick loader; fallback to stub."""
	try:
		# Prefer repo path
		import importlib.util
		import sys
		root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
		
		# Add repo root to sys.path for app imports
		if root not in sys.path:
			sys.path.insert(0, root)
			
		prod_loader_path = os.path.join(root, "data", "models_production", "quick_loader.py")
		print(f"🔍 Checking for production models at: {prod_loader_path}")
		print(f"🔍 File exists: {os.path.exists(prod_loader_path)}")
		
		if os.path.exists(prod_loader_path):
			spec = importlib.util.spec_from_file_location("quick_loader", prod_loader_path)
			assert spec and spec.loader
			quick_loader = importlib.util.module_from_spec(spec)
			sys.modules["quick_loader"] = quick_loader
			spec.loader.exec_module(quick_loader)
			result = quick_loader.predict_price(features)
			print(f"🎯 Production model result: {result}")
			return result
	except Exception as e:
		# Fall through to stub
		print(f"❌ Production model failed: {e}")
		pass

	# Stub result for demo
	return {
		"predicted_price": 42000.0,
		"trend": 1,
		"confidence": "medium",
		"model_name": "stub_linear",
		"metrics": {"r2": 0.0, "mae": 0.0},
		"note": "Using demo stub (no production models found)"
	}


# -------------------------------
# Demo data helpers (stubs)
# -------------------------------

def _fx_rate_usd_vnd() -> float:
	# Static rate for demo
	return float(os.getenv("FX_USD_VND", "24000"))


def _fmt_usd(x: float) -> str:
	return f"${x:,.2f}"


def _fmt_vnd(x: float) -> str:
	return f"{x:,.0f} đ"


def _fmt_pct(x: float) -> str:
	sign = "+" if x >= 0 else ""
	return f"{sign}{x:.2f}%"

def _now_str() -> str:
	return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def stub_market_snapshot(symbol: str) -> Dict[str, Any]:
	"""Return a deterministic demo market snapshot for a symbol.

	Fields:
	- price_usd, price_vnd, change_24h_pct, market_cap_usd, volume_24h_usd
	"""
	s = symbol.upper()
	# Base prices for a few known symbols, otherwise derive from hash
	base_prices = {
		"BTC": 68000.0,
		"ETH": 3500.0,
		"BNB": 500.0,
		"SOL": 180.0,
		"ADA": 0.45,
		"XRP": 0.6,
	}
	base = base_prices.get(s)
	if base is None:
		# Deterministic pseudo-random based on symbol hash
		h = abs(hash(s)) % 10000
		base = 10.0 + (h % 5000) / 10.0  # 10 .. ~5100
	# Change percent also deterministic but bounded
	chg = ((abs(hash(s + "chg")) % 6000) - 3000) / 100.0  # -30.00% .. +30.00%
	# Market cap and volume scale with price
	market_cap = base * 1_000_000  # demo value
	volume_24h = base * 100_000    # demo value
	fx = _fx_rate_usd_vnd()
	return {
		"symbol": s,
		"price_usd": float(base),
		"price_vnd": float(base * fx),
		"change_24h_pct": float(chg),
		"market_cap_usd": float(market_cap),
		"volume_24h_usd": float(volume_24h),
		"timestamp": _now_str(),
	}


def stub_ta_analysis(symbol: str) -> Dict[str, Any]:
	s = symbol.upper()
	base = stub_market_snapshot(s)
	# Simple heuristic for RSI/MA/Signal
	rsi = 30 + (hash(s) % 41)  # 30..70
	ma_signal = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"][hash(s) % 5]
	overall_signal = ["Buy", "Hold", "Sell"][hash(s[::-1]) % 3]
	support = round(base["price_usd"] * 0.97, 2)
	resistance = round(base["price_usd"] * 1.03, 2)
	stop = round(base["price_usd"] * 0.96, 2)
	target = round(base["price_usd"] * 1.05, 2)
	return {
		**base,
		"rsi": rsi,
		"ma_signal": ma_signal,
		"overall_signal": overall_signal,
		"support": support,
		"resistance": resistance,
		"stop": stop,
		"target": target,
		"timeframe": "24h Analysis",
	}


def stub_top_list(n: int = 10) -> List[Tuple[str, float, float]]:
	"""Return top N symbols with price and daily change (demo)."""
	universe = [
		"BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "TRX", "DOT", "MATIC",
		"AVAX", "LTC", "LINK", "ATOM", "NEAR"
	]
	data = []
	for sym in universe[:max(1, n)]:
		snap = stub_market_snapshot(sym)
		data.append((sym, snap["price_usd"], snap["change_24h_pct"]))
	return data


async def _binance_24h(symbol_usdt: str, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
	"""Fetch 24h ticker from Binance for SYMBOLUSDT (sync call offloaded to a thread)."""
	url = "https://api.binance.com/api/v3/ticker/24hr"
	def _do():
		return requests.get(url, params={"symbol": symbol_usdt}, timeout=timeout)
	try:
		resp = await asyncio.get_running_loop().run_in_executor(None, _do)
		if resp.status_code == 200:
			return resp.json()
	except Exception:
		return None
	return None


async def get_market_snapshot(symbol: str) -> Dict[str, Any]:
	"""Unified market snapshot using Binance realtime; fallback to stub.

	Returns keys: price_usd, price_vnd, change_24h_pct, market_cap_usd (None for Binance),
	volume_24h_usd, timestamp, symbol, source.
	"""
	s = symbol.upper()
	stats = await _binance_24h(f"{s}USDT")
	if stats and isinstance(stats, dict) and stats.get("lastPrice") is not None:
		try:
			price = float(stats.get("lastPrice") or 0)
			chg_pct = float(stats.get("priceChangePercent") or 0.0)
			vol_usd = float(stats.get("quoteVolume") or 0.0)
			fx = _fx_rate_usd_vnd()
			return {
				"symbol": s,
				"price_usd": price,
				"price_vnd": price * fx,
				"change_24h_pct": chg_pct,
				"market_cap_usd": None,
				"volume_24h_usd": vol_usd,
				"timestamp": _now_str(),
				"source": "binance",
			}
		except Exception:
			pass
	snap = stub_market_snapshot(s)
	snap["source"] = "stub"
	return snap


def create_bot() -> commands.Bot:
	intents = discord.Intents.default()
	intents.message_content = True
	bot = commands.Bot(command_prefix="!", intents=intents, help_command=None)

	# Global response tracking to prevent duplicates
	response_tracker = set()
	
	# Add a global check to prevent multiple instances responding
	@bot.check
	async def globally_block_dms(ctx):
		# Only respond in guilds (servers), not DMs
		return ctx.guild is not None

	# Command cooldown system to prevent duplicates
	command_cooldowns = {}
	
	# Track processed messages to prevent duplicate responses
	processed_messages = set()

	# Disk-based cross-instance claim directory
	CLAIMS_DIR = Path(os.path.dirname(__file__)) / ".claims"
	CLAIMS_DIR.mkdir(parents=True, exist_ok=True)

	def sweep_old_claims(max_age_sec: int = 3600, max_files: int = 2000) -> None:
		try:
			files = list(CLAIMS_DIR.glob("*.lock"))
			now = time.time()
			# Delete too old claims
			for p in files:
				try:
					if now - p.stat().st_mtime > max_age_sec:
						p.unlink(missing_ok=True)
				except Exception:
					pass
			# Bound total files
			files = list(CLAIMS_DIR.glob("*.lock"))
			if len(files) > max_files:
				for p in files[:-max_files]:
					try:
						p.unlink(missing_ok=True)
					except Exception:
						pass
		except Exception:
			pass

	def acquire_message_claim(message_id: int, command_name: str) -> bool:
		"""Acquire a cross-process claim via exclusive file create.
		Returns True if acquired; False if someone else already claimed.
		"""
		sweep_old_claims()
		fpath = CLAIMS_DIR / f"{message_id}.lock"
		try:
			with open(fpath, "x", encoding="utf-8") as f:
				f.write(json.dumps({
					"pid": os.getpid(),
					"ts": time.time(),
					"cmd": command_name,
				}))
			return True
		except FileExistsError:
			return False
		except Exception:
			return False

	async def should_respond_global(ctx, command_name: str) -> bool:
		"""Cross-instance dedup using a reaction claim + local checks.

		- First, try to add a 🤖 reaction to the triggering message. Only one
		  process with the same bot user can add it; the second one gets a 400
		  (already reacted) and we'll skip responding.
		- Then, fall back to local should_respond check to avoid re-entrancy.
		"""
		# Cross-process claim via lock file first (works without reaction perms)
		if not acquire_message_claim(ctx.message.id, command_name):
			print(f"🚫 CROSS-INSTANCE BLOCKED (file-claim): {ctx.message.id}_{command_name}")
			return False

		# Try cross-process claim via reaction unconditionally; ignore permission errors
		try:
			await ctx.message.add_reaction("🤖")
		except Exception:
			# If this fails (no permission or already reacted), we continue relying on file-claim
			pass

		# Local process-level duplicate/cooldown checks
		return await should_respond(ctx, command_name)

	async def is_on_cooldown(ctx, cooldown_seconds=3):
		"""Check if command is on cooldown for this user"""
		user_id = ctx.author.id
		command_name = ctx.command.name if ctx.command else ""
		key = f"{user_id}_{command_name}"
		
		now = time.time()
		if key in command_cooldowns:
			if now - command_cooldowns[key] < cooldown_seconds:
				return True
		
		command_cooldowns[key] = now
		return False
	
	async def is_duplicate_message(ctx):
		"""Check if we already processed this exact message"""
		message_key = f"{ctx.message.id}_{ctx.command.name if ctx.command else ''}"
		if message_key in processed_messages:
			return True
		processed_messages.add(message_key)
		
		# Clean old entries (keep last 1000)
		if len(processed_messages) > 1000:
			processed_messages.clear()
		
		return False
	
	async def should_respond(ctx, command_name):
		"""Local process duplicate guard based on message id."""
		request_key = str(ctx.message.id)
		if request_key in response_tracker:
			print(f"🚫 DUPLICATE BLOCKED: {request_key}")
			return False
		response_tracker.add(request_key)
		# bound memory
		if len(response_tracker) > 2000:
			response_tracker.clear()
		return True

	@bot.event
	async def on_ready():
		if bot.user is not None:
			print(f"🤖 Logged in as {bot.user} (ID: {bot.user.id})")
		else:
			print("🤖 Bot user is None (not logged in yet)")
		await bot.change_presence(activity=discord.Game(name="Crypto ML Demo"))

	# Note: We intentionally do NOT override on_message.
	# discord.py will route prefix commands automatically; a custom handler
	# risks double-processing if not carefully managed.

	@bot.command(name="ping")
	async def ping(ctx: commands.Context):
		await ctx.reply("🏓 Pong! Bot is alive.")

	@bot.command(name="help")
	async def help_cmd(ctx: commands.Context):
		help_text = (
			"🤖 Crypto ML Bot Commands:\n"
			"• !ping — Health check\n"
			"• !dudoan — Demo dự đoán giá (sử dụng model production nếu có)\n"
			"• !dudoan_json {json} — Dự đoán với JSON features tùy chỉnh\n"
			"• !price [SYMBOL] | !gia [SYMBOL] — Current price & quick prediction\n"
			"• !movers — Top gainers/losers 24h (demo)\n"
			"• !chart [SYMBOL] — Chart link or quick trend\n"
		)
		await ctx.reply(help_text)

	@commands.cooldown(1, 5, type=commands.BucketType.user)
	@commands.max_concurrency(1, per=commands.BucketType.channel, wait=False)
	@bot.command(name="dudoan", aliases=["predict"])  # keep old alias for compatibility
	async def dudoan(ctx: commands.Context, symbol: str = "BTC"):
		# Master duplicate check with cross-instance claim
		if not await should_respond_global(ctx, "dudoan"):
			return  # Silently ignore duplicate requests
		
		symbol = symbol.upper()
		snap = await get_market_snapshot(symbol)
		# Features scaled to training data range (mean ~$6k, not current BTC ~$117k)
		# Use a representative price from training distribution for better prediction
		base_price = 6000.0  # Close to training mean of $6,162
		features = {
			"open": base_price * 0.998,
			"high": base_price * 1.002,
			"low": base_price * 0.995,
			"close": base_price,
			"volume": 50_000,  # More realistic for training scale
			"ma_10": base_price * 0.996,
			"ma_50": base_price * 0.97,
			"volatility": 2.5,  # More realistic volatility
			"returns": 0.15,   # More realistic returns  
			"hour": datetime.now(timezone.utc).hour,
		}
		result = try_predict(features)
		
		# Debug: print what we got from try_predict
		print(f"🔍 try_predict returned: {result}")
		
		# Check if we got real model results or stub
		if result.get('model_name') == 'stub_linear':
			# Using stub - just use the stub price
			pred = float(result.get("predicted_price", snap["price_usd"]))
		else:
			# Real model used - scale the prediction appropriately
			pred = float(result.get("predicted_price", base_price))
			
			# Apply the prediction change percentage to actual current price
			pred_change_pct = (pred - base_price) / base_price
			pred = snap["price_usd"] * (1 + pred_change_pct)
		
		chg_pct = (pred - snap["price_usd"]) / max(1e-9, snap["price_usd"]) * 100.0
		trend_str = "Tăng" if (result.get("trend", 0) == 1 or chg_pct >= 0) else "Giảm"
		model_name = result.get('model_name') or 'Model (demo)'
		metrics = result.get('metrics') or {}
		r2 = metrics.get('r2')
		mae = metrics.get('mae')
		color = 0x9B59B6  # purple
		embed = discord.Embed(
			title=f"Dự đoán giá {symbol}",
			description="Dự báo 24 giờ tới",
			color=color,
			timestamp=datetime.now(timezone.utc)
		)
		embed.add_field(name="📊 Giá hiện tại", value=_fmt_usd(snap["price_usd"]), inline=True)
		embed.add_field(name="🎯 Giá dự đoán", value=_fmt_usd(pred), inline=True)
		embed.add_field(name="📈 Thay đổi dự kiến", value=_fmt_pct(chg_pct), inline=True)
		embed.add_field(name="🇻🇳 Giá VND hiện tại", value=_fmt_vnd(snap["price_vnd"]), inline=True)
		embed.add_field(name="💴 Giá VND dự đoán", value=_fmt_vnd(pred * _fx_rate_usd_vnd()), inline=True)
		embed.add_field(name="🔒 Độ tin cậy", value=str(result.get('confidence')).capitalize(), inline=True)
		embed.add_field(name="🕒 Thời gian", value="24 giờ", inline=True)
		extra = f"R²={r2:.3f}, MAE={mae:.2f}" if isinstance(r2,(int,float)) and isinstance(mae,(int,float)) else ""
		embed.add_field(name="🧠 Model", value=f"{model_name} {extra}".strip(), inline=True)
		embed.add_field(name="📈 Xu hướng", value=trend_str, inline=True)
		embed.set_footer(text="Đây chỉ là dự đoán, không phải lời khuyên đầu tư!")
		await ctx.reply(embed=embed)

	@commands.cooldown(1, 3, type=commands.BucketType.user)
	@commands.max_concurrency(1, per=commands.BucketType.channel, wait=False)
	@bot.command(name="dudoan_json", aliases=["predict_json"])  # keep old alias
	async def dudoan_json(ctx: commands.Context, *, payload: str):
		# Global dedup first
		if not await should_respond_global(ctx, "dudoan_json"):
			return
		try:
			features = json.loads(payload)
			if not isinstance(features, dict):
				raise ValueError("JSON payload must be an object with feature keys")
			result = try_predict(features)
			await ctx.reply(f"✅ Prediction: {json.dumps(result)}")
		except Exception as e:
			await ctx.reply(f"❌ Invalid JSON payload: {e}")

	@commands.cooldown(1, 3, type=commands.BucketType.user)
	@commands.max_concurrency(1, per=commands.BucketType.channel, wait=False)
	@bot.command(name="price", aliases=["gia"])  # !price BTC or !gia BTC
	async def price_cmd(ctx: commands.Context, symbol: str = "BTC"):
		# Master duplicate check (cross-instance)
		if not await should_respond_global(ctx, "price"):
			return  # Silently ignore
		
		symbol = symbol.upper()
		snap = await get_market_snapshot(symbol)
		color = 0x2ECC71  # green
		embed = discord.Embed(
			title=f"Giá {symbol}",
			color=color,
			timestamp=datetime.now(timezone.utc)
		)
		embed.add_field(name="💵 Giá hiện tại (USD)", value=_fmt_usd(snap["price_usd"]), inline=True)
		embed.add_field(name="🇻🇳 Giá VND", value=_fmt_vnd(snap["price_vnd"]), inline=True)
		embed.add_field(name="📉/📈 Thay đổi 24h", value=_fmt_pct(snap["change_24h_pct"]), inline=True)
		mc = snap.get("market_cap_usd")
		embed.add_field(name="🧮 Market Cap", value=("--" if mc in (None, 0) else _fmt_usd(mc)), inline=True)
		embed.add_field(name="📈 Volume 24h (USDT)", value=_fmt_usd(snap["volume_24h_usd"]), inline=True)
		label = os.getenv("BOT_INSTANCE") or f"{platform.node()}:{os.getpid()}"
		source = snap.get("source", "?")
		embed.set_footer(text=f"Cập nhật: {_now_str()} | Source: {source} | Instance: {label}")
		await ctx.reply(embed=embed)

	@commands.cooldown(1, 5, type=commands.BucketType.user)
	@commands.max_concurrency(1, per=commands.BucketType.channel, wait=False)
	@bot.command(name="movers")
	async def movers_cmd(ctx: commands.Context):
		# Global dedup + per-user cooldown
		if not await should_respond_global(ctx, "movers"):
			return
		if await is_on_cooldown(ctx, 5):  # 5 second cooldown
			return  # Silently ignore
			
		# Demo movers; in production compute from dataset cache
		gainers = [
			("BTC", +2.5), ("ETH", +1.8), ("BNB", +1.2)
		]
		losers = [
			("SOL", -3.1), ("ADA", -2.2), ("XRP", -1.7)
		]
		embed = discord.Embed(title="Top Movers 24h", color=0x3498DB, timestamp=datetime.now(timezone.utc))
		embed.add_field(name="🚀 Top tăng", value="\n".join([f"• {s}: {p:+.2f}%" for s,p in gainers]) or "--", inline=True)
		embed.add_field(name="📉 Top giảm", value="\n".join([f"• {s}: {p:+.2f}%" for s,p in losers]) or "--", inline=True)
		await ctx.reply(embed=embed)

	@commands.cooldown(1, 3, type=commands.BucketType.user)
	@commands.max_concurrency(1, per=commands.BucketType.channel, wait=False)
	@bot.command(name="chart")
	async def chart_cmd(ctx: commands.Context, symbol: str = "BTC"):
		# This command was reported to duplicate → add global dedup here
		if not await should_respond_global(ctx, "chart"):
			return
		symbol = symbol.upper()
		snap = await get_market_snapshot(symbol)
		# Simple TA around realtime price
		base_price = snap["price_usd"]
		rsi = 30 + (hash(symbol) % 41)
		ma_signal = ["Strong Buy", "Buy", "Hold", "Sell", "Strong Sell"][hash(symbol) % 5]
		overall_signal = ["Buy", "Hold", "Sell"][hash(symbol[::-1]) % 3]
		support = round(base_price * 0.97, 2)
		resistance = round(base_price * 1.03, 2)
		stop = round(base_price * 0.96, 2)
		target = round(base_price * 1.05, 2)
		color = 0xE74C3C if overall_signal == "Sell" else (0x2ECC71 if overall_signal == "Buy" else 0xF1C40F)
		embed = discord.Embed(
			title=f"Phân tích kỹ thuật {symbol}",
			description="Chỉ báo kỹ thuật và xu hướng",
			color=color,
			timestamp=datetime.now(timezone.utc)
		)
		embed.add_field(name="💵 Giá hiện tại", value=_fmt_usd(snap["price_usd"]), inline=True)
		embed.add_field(name="📈 RSI (14)", value=f"{float(rsi):.1f} - Neutral", inline=True)
		embed.add_field(name="📉 MA Signal", value=ma_signal, inline=True)
		embed.add_field(name="🛡️ Support Level", value=_fmt_usd(support), inline=True)
		embed.add_field(name="🎯 Resistance Level", value=_fmt_usd(resistance), inline=True)
		embed.add_field(name="📊 Overall Signal", value=overall_signal, inline=True)
		embed.add_field(name="📊 Volume 24h (USDT)", value=_fmt_usd(snap["volume_24h_usd"]), inline=True)
		embed.add_field(name="🎯 Price Targets", value=f"Stop: {_fmt_usd(stop)}\nTarget: {_fmt_usd(target)}", inline=True)
		embed.add_field(name="⏱️ Timeframe", value="24h Analysis", inline=True)
		link = f"https://www.tradingview.com/chart/?symbol={symbol}USD"
		label = os.getenv("BOT_INSTANCE") or f"{platform.node()}:{os.getpid()}"
		source = snap.get("source", "?")
		embed.set_footer(text=f"Đây chỉ là phân tích kỹ thuật, không phải lời khuyên đầu tư! | Source: {source} | Instance: {label}")
		embed.url = link
		await ctx.reply(embed=embed)

	@bot.command(name="ds")
	async def ds_cmd(ctx: commands.Context, top_n: int = 10):
		data = stub_top_list(top_n)
		color = 0x95A5A6
		embed = discord.Embed(title=f"Top {len(data)} Cryptocurrency", color=color, timestamp=datetime.now(timezone.utc))
		lines: List[str] = []
		for idx, (sym, price, chg) in enumerate(data, start=1):
			status_emoji = "✅" if price > 0 else "❌"
			chg_emoji = "📈" if chg >= 0 else "📉"
			lines.append(f"{idx}. {sym}\n{_fmt_usd(price)}  {chg_emoji} {_fmt_pct(chg)} {status_emoji}")
		embed.description = "\n\n".join(lines)
		label = os.getenv("BOT_INSTANCE") or f"{platform.node()}:{os.getpid()}"
		embed.set_footer(text=f"Cập nhật: {_now_str()} | Instance: {label}")
		await ctx.reply(embed=embed)

	return bot


def main() -> int:
	# Single-instance guard using a lock file; tolerate stale locks
	lock_path = os.path.join(os.path.dirname(__file__), ".bot.lock")

	def remove_lock():
		try:
			if os.path.exists(lock_path):
				os.remove(lock_path)
		except Exception:
			pass

	if os.path.exists(lock_path):
		try:
			with open(lock_path, "r") as f:
				pid_str = f.read().strip()
			other_pid = int(pid_str) if pid_str.isdigit() else None
		except Exception:
			other_pid = None

		# Conservative: assume running unless we can prove not running
		other_running = True
		if other_pid is None:
			other_running = True
		else:
			if psutil:
				try:
					if not psutil.pid_exists(other_pid):
						other_running = False
					else:
						p = psutil.Process(other_pid)
						if not p.is_running():
							other_running = False
						elif other_pid == os.getpid():
							other_running = False
				except Exception:
					other_running = True
			else:
				# Without psutil, err on the side of caution
				other_running = True

		if other_running:
			print(f"❌ Another bot instance seems to be running (PID: {other_pid})")
			print("Stop that process or delete app/.bot.lock if you're sure it's stale.")
			return 3
		else:
			# Stale lock, remove it
			remove_lock()

	# Acquire lock
	try:
		with open(lock_path, "w") as f:
			f.write(str(os.getpid()))
		atexit.register(remove_lock)
	except Exception as e:
		print(f"⚠️ Could not create lock file: {e}")

	# Optional: soft warning if another instance appears via psutil scan
	if psutil:
		try:
			current_pid = os.getpid()
			suspects = []
			for proc in psutil.process_iter(["pid", "name", "cmdline"]):
				if proc.info["pid"] == current_pid:
					continue
				name = (proc.info.get("name") or "").lower()
				cmdline = " ".join(proc.info.get("cmdline") or [])
				if ("python" in name) and ("bot.py" in cmdline):
					suspects.append(proc.info["pid"])
			if suspects:
				print(f"⚠️ Warning: Other potential bot processes detected (PIDs: {suspects}).")
				print("Proceeding due to lock file guard; ensure only one instance is intended.")
		except Exception as e:
			print(f"⚠️ psutil scan failed: {e}")
	else:
		print("⚠️ psutil not available; relying on lock file only.")

	token = read_bot_token()
	if not token:
		print("❌ BOT_TOKEN not provided. Set env BOT_TOKEN or create token.txt with your token.")
		return 1

	print("🔍 Starting single bot instance...")
	bot = create_bot()
	try:
		bot.run(token)
	except Exception as e:
		print(f"❌ Failed to start Discord bot: {e}")
		return 2
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

