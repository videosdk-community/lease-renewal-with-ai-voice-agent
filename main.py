
import os
from videosdk.agents import Agent, AgentSession, Pipeline, JobContext, RoomOptions, WorkerJob
from videosdk.plugins.silero import SileroVAD
from videosdk.plugins.turn_detector import TurnDetector, pre_download_model
from videosdk.plugins.deepgram import DeepgramSTT
from videosdk.plugins.openai import OpenAILLM
from videosdk.plugins.elevenlabs import ElevenLabsTTS

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", handlers=[logging.StreamHandler()])

# Pre-downloading the Turn Detector model
pre_download_model()

class MyVoiceAgent(Agent):
    def __init__(self):
        super().__init__(instructions="""You are a knowledgeable and efficient lease renewal specialist. Your primary role is to assist tenants in understanding and completing the lease renewal process. You should provide clear information on lease terms, renewal options, and any changes in policies or rental rates. Maintain a courteous and professional tone, ensuring that tenants feel valued and informed. Be proactive in addressing potential questions or concerns, such as changes in lease duration, rent payment options, and maintenance requests. Offer guidance on how to digitally sign lease documents and provide contact information for further assistance if needed. Do not handle personal financial information directly; instead, direct tenants to the secure online portal for any financial transactions.""")
    async def on_enter(self): await self.session.say("Hello! How can I help you today regarding lease renewal with ai voice agent?")
    async def on_exit(self): await self.session.say("Goodbye!")

async def start_session(context: JobContext):
    # Create agent
    agent = MyVoiceAgent()

    # Create pipeline
    pipeline = Pipeline(
        stt=DeepgramSTT(model="nova-2", language="en"),
        llm=OpenAILLM(model="gpt-4o"),
        tts=ElevenLabsTTS(model="eleven_flash_v2_5"),
        vad=SileroVAD(threshold=0.35),
        turn_detector=TurnDetector(threshold=0.8)
    )

    session = AgentSession(
        agent=agent,
        pipeline=pipeline
    )

    await session.start(wait_for_participant=True, run_until_shutdown=True)

def make_context() -> JobContext:
    room_options = RoomOptions(
     #  room_id="<room_id>",  # Set to join a pre-created room; omit to auto-create
        name="VideoSDK Cascaded Agent for lease renewal with ai voice agent",
        playground=True
    )

    return JobContext(room_options=room_options)

if __name__ == "__main__":
    job = WorkerJob(entrypoint=start_session, jobctx=make_context)
    job.start()
